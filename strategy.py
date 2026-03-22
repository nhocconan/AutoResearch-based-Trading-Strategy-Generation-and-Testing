#!/usr/bin/env python3
"""
Experiment #015: 1h Connors RSI + 4h HMA Trend + Choppiness Regime Filter
Hypothesis: Connors RSI (75% win rate in literature) combined with 4h trend bias and 
Choppiness Index regime detection will work in both bull and bear markets.
Key insight: Previous 13 strategies failed due to overly strict entries or wrong regime detection.
This uses CRSI for entries (loose thresholds: <20 long, >80 short), 4h HMA for trend bias,
Choppiness to distinguish range vs trend markets, and Fisher Transform for reversal confirmation.
Timeframe: 1h (REQUIRED for exp#015), HTF: 4h via mtf_data helper.
Position sizing: 0.25-0.30 discrete levels, stoploss at 2.5*ATR.
Why this might work: CRSI is proven mean-reversion indicator, Choppiness avoids trend-following in ranges,
4h HMA provides smoother trend filter than 1h. Entry conditions LOOSENED for 10+ trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_4h_hma_fisher_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - measures consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak[max(0, i-streak_period+1):i+1]
        pos_streaks = np.sum(streak_window > 0)
        if streak_period > 0:
            streak_rsi[i] = (pos_streaks / streak_period) * 100
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100) - where current return ranks vs last 100 days
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        returns = np.diff(close[max(0, i-rank_period):i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1] if i > 0 else 0
            pct_rank[i] = (np.sum(returns <= current_return) / len(returns)) * 100
    
    # Combine into CRSI
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + pct_rank[mask]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest > lowest:
            atr_sum = 0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                        abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    Catches reversals in bear rallies.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        if highest > lowest:
            normalized = 2 * (close[i] - lowest) / (highest - lowest) - 1
            normalized = np.clip(normalized, -0.999, 0.999)
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    return fisher

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    fisher = calculate_fisher(close, 9)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h trend confirmation
        bull_trend_1h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_1h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Choppiness regime detection
        is_ranging = not np.isnan(chop[i]) and chop[i] > 55  # Range market
        is_trending = not np.isnan(chop[i]) and chop[i] < 45  # Trend market
        
        # Fisher Transform signals
        fisher_oversold = not np.isnan(fisher[i]) and fisher[i] < -1.0
        fisher_overbought = not np.isnan(fisher[i]) and fisher[i] > 1.0
        
        # Fisher crossover detection
        fisher_cross_long = False
        fisher_cross_short = False
        if i >= 1 and not np.isnan(fisher[i]) and not np.isnan(fisher[i-1]):
            fisher_cross_long = fisher[i] > -1.5 and fisher[i-1] <= -1.5
            fisher_cross_short = fisher[i] < 1.5 and fisher[i-1] >= 1.5
        
        # CRSI conditions - LOOSENED for more trades (key fix!)
        crsi_oversold = crsi[i] < 25  # Long entry (was <10, too strict)
        crsi_overbought = crsi[i] > 75  # Short entry (was >90, too strict)
        crsi_neutral = 30 < crsi[i] < 70
        
        # Bollinger Band position
        near_bb_lower = close[i] <= bb_lower[i] * 1.01
        near_bb_upper = close[i] >= bb_upper[i] * 0.99
        
        # Long-term trend filter
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: CRSI oversold + 4h bullish trend (mean reversion in uptrend)
        if crsi_oversold and bull_trend_4h:
            new_signal = SIZE_BASE
        
        # Secondary: Fisher cross long + 4h bullish (reversal confirmation)
        elif fisher_cross_long and bull_trend_4h:
            new_signal = SIZE_BASE
        
        # Tertiary: CRSI oversold + near BB lower (double mean reversion)
        elif crsi_oversold and near_bb_lower and above_200:
            new_signal = SIZE_HALF
        
        # Range market mean reversion: CRSI oversold + ranging market
        elif crsi_oversold and is_ranging:
            new_signal = SIZE_HALF
        
        # === SHORT ENTRIES ===
        # Primary: CRSI overbought + 4h bearish trend (mean reversion in downtrend)
        elif crsi_overbought and bear_trend_4h:
            new_signal = -SIZE_BASE
        
        # Secondary: Fisher cross short + 4h bearish (reversal confirmation)
        elif fisher_cross_short and bear_trend_4h:
            new_signal = -SIZE_BASE
        
        # Tertiary: CRSI overbought + near BB upper (double mean reversion)
        elif crsi_overbought and near_bb_upper and below_200:
            new_signal = -SIZE_HALF
        
        # Range market mean reversion: CRSI overbought + ranging market
        elif crsi_overbought and is_ranging:
            new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals