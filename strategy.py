#!/usr/bin/env python3
"""
Experiment #034: 4h Connors RSI Mean Reversion with 1d HMA Regime Filter
Hypothesis: Connors RSI (CRSI) provides superior mean reversion signals vs standard RSI.
Combined with 1d HMA for trend bias and Bollinger Band Width for regime detection.
Key insight: BTC/ETH fail on pure trend strategies but excel at mean reversion in range markets.
CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 captures short-term oversold/overbought better.
Timeframe: 4h (REQUIRED for exp#034), HTF: 1d via mtf_data helper.
Position sizing: 0.20-0.30 discrete, ATR stoploss at 2.5*ATR.
Why this might work: CRSI has 75% win rate in backtests, works through 2022 crash (mean revert at bottom).
Must generate 10+ trades on train, 3+ on test - CRSI thresholds loosened (15/85 not 10/90).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_1d_hma_bb_regime_v1"
timeframe = "4h"
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
    Original formula from Connors & Alvarez (2009).
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: Short-term RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI
    # Count consecutive up/down days, then calculate RSI on streak values
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, streak_period)
    
    # Component 3: Percent Rank of 1-day price changes
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = delta[i-rank_period+1:i+1]
        if len(window) > 0:
            rank = np.sum(window < delta[i])
            percent_rank[i] = rank / len(window) * 100
    
    # Combine components
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / (sma + 1e-10)
    return upper, lower, sma, bandwidth

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_percentile_rank(values, window=100):
    """Calculate rolling percentile rank."""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    for i in range(window, n):
        window_vals = values[i-window+1:i+1]
        if len(window_vals) > 0:
            pr[i] = np.sum(window_vals < values[i]) / len(window_vals) * 100
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    
    # Connors RSI for mean reversion entries
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Standard RSI for confirmation
    rsi_14 = calculate_rsi(close, 14)
    
    # Bollinger Bands for regime detection
    bb_upper, bb_lower, bb_sma, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    
    # BB Width percentile for regime (low = trending, high = ranging)
    bb_width_pr = calculate_percentile_rank(bb_width, window=100)
    
    # EMAs for trend confirmation
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pr[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - main regime filter
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # Regime detection via BB Width percentile
        # Low BW percentile (<30) = trending market (use trend-follow entries)
        # High BW percentile (>70) = ranging market (use mean-revert entries)
        is_trending = bb_width_pr[i] < 30
        is_ranging = bb_width_pr[i] > 50
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # CRSI conditions - LOOSENED for more trades (15/85 instead of 10/90)
        crsi_oversold = crsi[i] < 20  # Mean reversion long signal
        crsi_overbought = crsi[i] > 80  # Mean reversion short signal
        
        # RSI confirmation
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # Price position relative to BB
        near_bb_lower = close[i] <= bb_lower[i] * 1.01
        near_bb_upper = close[i] >= bb_upper[i] * 0.99
        
        # EMA alignment for trend confirmation
        ema_bullish = ema_21[i] > ema_50[i] if not np.isnan(ema_50[i]) else False
        ema_bearish = ema_21[i] < ema_50[i] if not np.isnan(ema_50[i]) else False
        
        # Price pullback to EMA21 in trend
        pullback_to_ema_long = close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.97
        pullback_to_ema_short = close[i] >= ema_21[i] * 0.99 and close[i] <= ema_21[i] * 1.03
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        if bull_trend_1d:
            # Primary: CRSI oversold in ranging market (mean reversion)
            if is_ranging and crsi_oversold and above_200:
                new_signal = SIZE_BASE
            
            # Secondary: CRSI oversold + price at BB lower in trending market
            elif is_trending and crsi_oversold and near_bb_lower and ema_bullish:
                new_signal = SIZE_BASE
            
            # Tertiary: Pullback to EMA21 with RSI confirmation
            elif pullback_to_ema_long and rsi_oversold and bull_trend_1d:
                new_signal = SIZE_HALF
            
            # Momentum: Price above EMA200 with CRSI recovering
            elif above_200 and crsi[i] > 30 and crsi[i-1] < 25 if i > 0 else False:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES ===
        elif bear_trend_1d:
            # Primary: CRSI overbought in ranging market (mean reversion)
            if is_ranging and crsi_overbought and below_200:
                new_signal = -SIZE_BASE
            
            # Secondary: CRSI overbought + price at BB upper in trending market
            elif is_trending and crsi_overbought and near_bb_upper and ema_bearish:
                new_signal = -SIZE_BASE
            
            # Tertiary: Bounce to EMA21 with RSI confirmation
            elif pullback_to_ema_short and rsi_overbought and bear_trend_1d:
                new_signal = -SIZE_HALF
            
            # Momentum: Price below EMA200 with CRSI declining
            elif below_200 and crsi[i] < 70 and crsi[i-1] > 75 if i > 0 else False:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
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