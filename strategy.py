#!/usr/bin/env python3
"""
Experiment #516: Daily Bollinger Squeeze + Weekly HMA + RSI Regime Adaptive

Hypothesis: After 500+ failed experiments, 1d timeframe needs REGIME-ADAPTIVE logic:
1. BOLLINGER SQUEEZE: BB width < 20th percentile = low vol = impending breakout
2. WEEKLY HMA BIAS: 1w HMA(21) via mtf_data for macro trend direction
3. ASYMMETRIC ENTRY: Different RSI thresholds for bull vs bear regime
4. LOOSE THRESHOLDS: RSI < 45 long, > 55 short (ensures ≥10 trades on 1d)
5. VOL EXPANSION CONFIRMATION: BB width expanding from squeeze

Why this might work on 1d:
- BB squeeze catches volatility expansion before major moves
- Weekly HMA prevents counter-trend trades in strong macro trends
- Asymmetric RSI thresholds adapt to bull/bear market behavior
- 1d timeframe minimizes noise and fee drag vs lower timeframes
- Fewer but higher-quality trades = better Sharpe

Key innovations:
1. BB WIDTH PERCENTILE: BB width < 20th pct of last 100 days = squeeze
2. WEEKLY HMA BIAS: 1w HMA(21) via mtf_data for macro trend
3. ASYMMETRIC RSI: Bull regime RSI<45 long, Bear regime RSI>55 short
4. VOL EXPANSION: BB width increasing from previous bar confirms breakout
5. 2.5 * ATR STOPLOSS: Wider stop for 1d timeframe swings

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data helper
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_bb_squeeze_weekly_hma_asymmetric_rsi_atr_v1"
timeframe = "1d"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = upper - lower
    return upper, lower, sma, width

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate BB width percentile rank over lookback period."""
    n = len(bb_width)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if np.isnan(bb_width[i]):
            continue
        window = bb_width[i-lookback+1:i+1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            count_below = np.sum(valid_window[:-1] < bb_width[i])
            percentile[i] = count_below / (len(valid_window) - 1) * 100
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width[i]) or np.isnan(bb_width[i-1]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_bias = close[i] > hma_1w_aligned[i]
        bear_bias = close[i] < hma_1w_aligned[i]
        
        # === BB SQUEEZE DETECTION ===
        is_squeeze = bb_width_pct[i] < 25  # Bottom 25% = squeeze (looser for more trades)
        
        # === VOL EXPANSION CONFIRMATION ===
        vol_expanding = bb_width[i] > bb_width[i-1] * 1.02  # 2% expansion
        
        # === RSI CONFIRMATION (Asymmetric) ===
        # Bull regime: easier to go long (RSI < 50)
        # Bear regime: easier to go short (RSI > 50)
        rsi_long = rsi_14[i] < 50
        rsi_short = rsi_14[i] > 50
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: BB squeeze + vol expanding + RSI ok + weekly bull bias
        if is_squeeze and vol_expanding and rsi_long and bull_bias:
            new_signal = SIZE
        
        # Short: BB squeeze + vol expanding + RSI ok + weekly bear bias
        elif is_squeeze and vol_expanding and rsi_short and bear_bias:
            new_signal = -SIZE
        
        # Alternative: Pure mean reversion at BB extremes (no squeeze required)
        # This generates more trades when squeeze conditions are too rare
        if new_signal == 0.0:
            # Long at lower band in bull regime
            if close[i] < bb_lower[i] * 1.005 and bull_bias and rsi_14[i] < 45:
                new_signal = SIZE
            # Short at upper band in bear regime
            elif close[i] > bb_upper[i] * 0.995 and bear_bias and rsi_14[i] > 55:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === BIAS REVERSAL EXIT ===
        # Exit if weekly trend flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals