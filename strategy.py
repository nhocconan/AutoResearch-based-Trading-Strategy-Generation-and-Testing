#!/usr/bin/env python3
"""
Experiment #582: 1d Weekly HMA Trend with RSI Pullback and Volatility Filter

Hypothesis: After 500+ failed experiments, the key insight is:
1. 1d timeframe captures multi-week trends without intraday noise
2. Weekly HMA (21) provides strong trend bias - proven in best strategies
3. RSI(14) pullback entries work better than breakouts in crypto
4. ATR volatility filter avoids entering during extreme volatility spikes
5. Conservative sizing (0.25) protects against 2022-style crashes
6. Simple logic = more trades = statistical significance

Why this should work on 1d:
- 1d has 1 bar/day = ~365 bars/year = quality over quantity
- Weekly HMA trend bias prevents counter-trend entries (major failure mode)
- RSI(14) < 35 in uptrend = quality pullback entry (not oversold crash)
- ATR ratio filter avoids panic entries (ATR(7)/ATR(21) < 1.5)
- 2*ATR stoploss protects against sustained moves against position
- Should generate 15-30 trades/year = statistical significance

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (max 0.40)
Stoploss: 2*ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_weekly_hma_rsi_pullback_vol_filter_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

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
    atr_7 = calculate_atr(high, low, close, 7)
    atr_21 = calculate_atr(high, low, close, 21)
    rsi_14 = calculate_rsi(close, 14)
    
    # ATR ratio for volatility filter
    atr_ratio = atr_7 / np.where(atr_21 == 0, np.inf, atr_21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_bias = close[i] > hma_1w_aligned[i]
        bear_bias = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY FILTER ===
        # Avoid entering during extreme volatility spikes (panic/crash)
        vol_normal = atr_ratio[i] < 1.8  # Not in extreme vol spike
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pullback in uptrend (not crash, just pullback)
        rsi_pullback_long = rsi_14[i] < 45 and rsi_14[i] > 25
        # Short: RSI rally in downtrend
        rsi_rally_short = rsi_14[i] > 55 and rsi_14[i] < 75
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: RSI pullback + weekly bullish bias + normal volatility
        if rsi_pullback_long and bull_bias and vol_normal:
            new_signal = SIZE
        
        # Short: RSI rally + weekly bearish bias + normal volatility
        elif rsi_rally_short and bear_bias and vol_normal:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if weekly HMA flips against position
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
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals