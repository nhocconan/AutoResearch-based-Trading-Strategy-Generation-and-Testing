#!/usr/bin/env python3
"""
Experiment #183: 1h Volatility Spike Mean Reversion + 4h HMA Trend Filter + BB Regime

Hypothesis: 1h timeframe is ideal for volatility spike mean reversion. After panic
sell-offs (vol spike), price tends to revert to mean within 12-48 hours. Combined
with 4h HMA trend filter to avoid counter-trend trades, and Bollinger Band regime
to identify oversold/overbought conditions. This should work better than pure
trend-following on 1h which has failed repeatedly (Sharpe -1.375, -0.061).

Why 1h might work for vol spike reversion:
- 1h captures panic moves faster than 4h/12h
- Vol spike (ATR7/ATR30 > 2.0) signals exhaustion, not continuation
- BB(20, 2.5) extremes = oversold/overbought on 1h
- 4h HMA filter prevents fighting major trend
- More trades than 4h/12h strategies (need >=10 trades per symbol)

Learning from failures:
- #171 (1h KAMA): Sharpe=-1.375 - trend following whipsaws on 1h
- #177 (1h KAMA+Donchian): Sharpe=-0.061 - breakout logic fails on 1h noise
- #175 (15m pullback): Sharpe=-4.690 - too much noise on lower TF
- Mean reversion CAN work on 1h IF filtered by vol spike + HTF trend
- Need flexible entry conditions to ensure trade count

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h HMA via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_spike_4h_hma_bb_meanrev_v1"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.values, lower.values, sma.values

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
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / (rolling_std + 1e-10)
    return zscore.values

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
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    zscore = calculate_zscore(close, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR(7) / ATR(30) > 2.0 = volatility spike (panic/exhaustion)
        vol_spike_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = vol_spike_ratio > 2.0
        
        # === 4H TREND BIAS ===
        # Price > 4h HMA = bullish bias (only take longs or wait)
        # Price < 4h HMA = bearish bias (only take shorts or wait)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === BOLLINGER BAND EXTREMES ===
        # Price at lower BB = oversold (long opportunity)
        # Price at upper BB = overbought (short opportunity)
        at_lower_bb = close[i] <= bb_lower[i]
        at_upper_bb = close[i] >= bb_upper[i]
        
        # === RSI EXTREMES ===
        # RSI < 30 = oversold
        # RSI > 70 = overbought
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # === Z-SCORE EXTREMES ===
        # Z-score < -2 = significantly below mean
        # Z-score > +2 = significantly above mean
        zscore_low = zscore[i] < -2.0
        zscore_high = zscore[i] > 2.0
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: Vol spike OR (at lower BB + RSI oversold) + 4h trend bullish
        # More flexible to ensure enough trades
        long_conditions = 0
        if vol_spike and at_lower_bb:
            long_conditions += 2
        elif vol_spike:
            long_conditions += 1
        elif at_lower_bb and rsi_oversold:
            long_conditions += 1
        elif at_lower_bb and zscore_low:
            long_conditions += 1
        
        if bull_trend_4h and long_conditions >= 1:
            new_signal = SIZE_BASE
        
        # Short: Vol spike OR (at upper BB + RSI overbought) + 4h trend bearish
        short_conditions = 0
        if vol_spike and at_upper_bb:
            short_conditions += 2
        elif vol_spike:
            short_conditions += 1
        elif at_upper_bb and rsi_overbought:
            short_conditions += 1
        elif at_upper_bb and zscore_high:
            short_conditions += 1
        
        if bear_trend_4h and short_conditions >= 1:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr_14[i]
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
            # else: maintaining same position direction
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