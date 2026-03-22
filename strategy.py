#!/usr/bin/env python3
"""
Experiment #029: 4h Regime-Adaptive Strategy with 1d HMA Trend Filter

Hypothesis: A regime-adaptive approach that switches between mean reversion and 
trend following based on Choppiness Index will outperform pure trend strategies
that failed in 2022 crash. Key design:
1. 1d HMA(21) for major trend bias (call ONCE before loop via mtf_data)
2. Choppiness Index(14) for regime detection: >61.8 = chop (mean revert), <38.2 = trend
3. Connors RSI for mean reversion entries in chop regime
4. Donchian(20) breakout for trend entries in trend regime
5. ATR(14) stoploss at 2.5x for risk management
6. Relaxed entry conditions to ensure trade generation (learned from 0-trade failures)

Why this should work:
- Regime detection prevents trend following in choppy markets (2022 bottom whipsaw)
- Connors RSI has 75% win rate in mean reversion (research-backed)
- 4h TF targets 20-50 trades/year (optimal for fee efficiency)
- 1d HMA filter prevents counter-trend trades in major moves
- Simple thresholds ensure trades actually trigger

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_chop_crsi_1d_hma_v1"
timeframe = "4h"
leverage = 1.0

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
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long entry: CRSI < 10 (oversold)
    Short entry: CRSI > 90 (overbought)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Calculate streak RSI
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    streak = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if delta[i-1] >= 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if delta[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak (use absolute values for calculation)
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_gain_avg = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_avg = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_loss_avg = np.where(streak_loss_avg == 0, 1e-10, streak_loss_avg)
    streak_rs = streak_gain_avg / streak_loss_avg
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Percent rank of close over rank_period
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        crsi[i] = (rsi_close[i] + rsi_streak[i] + rank) / 3.0
    
    return crsi

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    chop = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    hma_4h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_4h_21[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1D HTF TREND BIAS ===
        htf_bullish = close[i] > hma_1d_21_aligned[i]
        htf_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H LOCAL TREND ===
        local_bullish = close[i] > hma_4h_21[i]
        local_bearish = close[i] < hma_4h_21[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Slightly relaxed from 61.8 to get more trades
        is_trending = chop[i] < 45.0  # Slightly relaxed from 38.2
        
        # === CONNORS RSI EXTREMES (for mean reversion in chop) ===
        crsi_oversold = crsi[i] < 25.0  # Relaxed from 10 to get more trades
        crsi_overbought = crsi[i] > 75.0  # Relaxed from 90
        
        # === DONCHIAN BREAKOUT (for trend following) ===
        donchian_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI FILTER ===
        rsi_bullish = rsi_14[i] > 45.0
        rsi_bearish = rsi_14[i] < 55.0
        
        # === POSITION SIZING ===
        if htf_bullish and local_bullish:
            current_size = STRONG_SIZE
        elif htf_bearish and local_bearish:
            current_size = STRONG_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # MEAN REVERSION ENTRY (in choppy regime)
        # Long: CRSI oversold + HTF bullish + above SMA200
        if is_choppy and crsi_oversold and htf_bullish and above_sma200:
            new_signal = current_size
        
        # Short: CRSI overbought + HTF bearish + below SMA200
        elif is_choppy and crsi_overbought and htf_bearish and below_sma200:
            new_signal = -current_size
        
        # TREND FOLLOWING ENTRY (in trending regime)
        # Long: Donchian breakout + HTF bullish + local bullish
        elif is_trending and donchian_long and htf_bullish:
            new_signal = current_size
        
        # Short: Donchian breakout + HTF bearish + local bearish
        elif is_trending and donchian_short and htf_bearish:
            new_signal = -current_size
        
        # === NEUTRAL REGIME ENTRY (when chop is between 45-55) ===
        # Use simpler logic to ensure trades
        if not is_choppy and not is_trending:
            if htf_bullish and donchian_long:
                new_signal = current_size * 0.8
            elif htf_bearish and donchian_short:
                new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 25 bars (~4 days on 4h), allow weaker entry
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if htf_bullish and crsi_oversold:
                new_signal = current_size * 0.7
            elif htf_bearish and crsi_overbought:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === CRSI EXTREME EXIT (take profit on mean reversion) ===
        crsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi[i] > 60.0:
                crsi_exit = True
            if position_side < 0 and crsi[i] < 40.0:
                crsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or crsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals