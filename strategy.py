#!/usr/bin/env python3
"""
Experiment #596: 30m Trend-Following with 4h HMA + RSI Pullback + Z-Score Filter

Hypothesis: After 520+ failures, the winning pattern is SIMPLE trend-following 
with pullback entries, NOT complex regime detection. The current best strategy 
(Sharpe=0.676) uses 4h trend + adaptive logic.

Key insights from failures:
- Complex regime detection (CHOP, multiple modes) = overfitting
- Too many filters = 0 trades or late entries
- RSI pullback to trend works better than breakout strategies
- Z-score filter helps avoid entering at extremes

This strategy:
1. 4h HMA(21) = primary trend direction (call ONCE before loop)
2. 30m RSI(14) pullback = entry trigger (35-45 for long, 55-65 for short)
3. Z-score(20) filter = avoid extremes (|z| < 1.5)
4. Asymmetric sizing: 0.25 long, 0.20 short (bear market bias)
5. Stoploss: 2.5 * ATR(14)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_trend_4h_hma_rsi_pullback_zscore_atr_v1"
timeframe = "30m"
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

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std.replace(0, np.inf)
    return zscore.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    zscore_20 = calculate_zscore(close, 20)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_LONG = 0.28
    SIZE_SHORT = 0.22  # Slightly smaller for shorts (bear market bias)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(zscore_20[i]) or np.isnan(ema_21[i]):
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # === 30M RSI PULLBACK ENTRY ===
        # Long: RSI pulled back to 35-50 in uptrend
        rsi_pullback_long = 35 <= rsi_14[i] <= 50
        # Short: RSI pulled back to 50-65 in downtrend
        rsi_pullback_short = 50 <= rsi_14[i] <= 65
        
        # === Z-SCORE FILTER (avoid extremes) ===
        zscore_ok_long = zscore_20[i] < 1.0  # Not overbought
        zscore_ok_short = zscore_20[i] > -1.0  # Not oversold
        
        # === EMA CONFIRMATION ===
        ema_bull = close[i] > ema_21[i]
        ema_bear = close[i] < ema_21[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long entry: 4h bullish + RSI pullback + Z-score ok + EMA confirmation
        if bull_trend and rsi_pullback_long and zscore_ok_long and ema_bull:
            new_signal = SIZE_LONG
        
        # Short entry: 4h bearish + RSI pullback + Z-score ok + EMA confirmation
        elif bear_trend and rsi_pullback_short and zscore_ok_short and ema_bear:
            new_signal = -SIZE_SHORT
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend flips bearish
            if position_side > 0 and bear_trend and rsi_14[i] > 60:
                trend_reversal = True
            # Exit short if 4h trend flips bullish
            if position_side < 0 and bull_trend and rsi_14[i] < 40:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
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
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals