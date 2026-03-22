#!/usr/bin/env python3
"""
Experiment #219: 1h Fisher Transform + 4h HMA Trend + Volume Confirmation + ATR Stop

Hypothesis: 1h timeframe with Fisher Transform entries can catch reversals faster than RSI,
while 4h HMA provides stable trend bias. Volume confirmation filters false breakouts.
This should work better than RSI/MACD approaches that failed on 1h recently.

Why this might work:
- Fisher Transform (Ehlers) converts price to Gaussian for clearer turning points
- 4h HMA trend filter prevents counter-trend trades (proven in best strategies)
- Volume spike confirmation (1.5x average) validates breakout authenticity
- 1h captures moves faster than 4h but with less noise than 15m/30m
- Conservative sizing (0.25) with 2.5*ATR stop controls drawdown

Learning from failures:
- #207 (1h RSI mean rev): Sharpe=-9.084 - mean reversion fails on 1h
- #211 (15m MACD): Sharpe=-2.490 - too much noise on lower TF
- #213 (1h MACD): Sharpe=-0.864 - MACD laggy without proper filter
- Best strategy uses 4h KAMA + 1d HMA + ADX (trend-following works)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_vol_atr_v1"
timeframe = "1h"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price to Gaussian distribution for clearer turning points.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_prev = np.zeros(n)
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            fisher[i] = fisher_prev[i-1]
            fisher_prev[i] = fisher[i]
            continue
        
        normalized = (hl2 - lowest) / range_val
        x = 0.66 * (normalized - 0.5) + 0.67 * fisher_prev[i-1]
        x = np.clip(x, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_spike(volume, period=20):
    """Calculate volume spike ratio (current / average)."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, 9)
    vol_ratio = calculate_volume_spike(volume, 20)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(vol_ratio[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        # Volume spike > 1.3x average confirms breakout validity
        vol_confirmed = vol_ratio[i] > 1.3
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.0 from below (oversold reversal)
        # Short: Fisher crosses below +1.0 from above (overbought reversal)
        fisher_long_cross = fisher[i] > -1.0 and fisher_prev[i] <= -1.0
        fisher_short_cross = fisher[i] < 1.0 and fisher_prev[i] >= 1.0
        
        # Alternative: Fisher direction with extreme levels
        fisher_rising = fisher[i] > fisher_prev[i]
        fisher_falling = fisher[i] < fisher_prev[i]
        fisher_oversold = fisher[i] < -0.5
        fisher_overbought = fisher[i] > 0.5
        
        # === RSI CONFIRMATION ===
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + Fisher reversal + volume confirmation OR RSI support
        if bull_trend_4h:
            if fisher_long_cross and vol_confirmed:
                new_signal = SIZE_BASE
            elif fisher_oversold and fisher_rising and rsi_bullish:
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + Fisher reversal + volume confirmation OR RSI support
        if bear_trend_4h:
            if fisher_short_cross and vol_confirmed:
                new_signal = -SIZE_BASE
            elif fisher_overbought and fisher_falling and rsi_bearish:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals