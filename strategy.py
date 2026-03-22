#!/usr/bin/env python3
"""
Experiment #020: 1h Fisher Transform Entries with 4h/12h HMA Trend

Hypothesis: Previous 1h strategies failed due to over-filtering (session + volume = 0 trades).
This strategy simplifies to proven pattern: HTF trend + reversal indicator for timing.

Key components:
1. 4h HMA(16/48) crossover for trend direction (proven in current best baseline)
2. 12h HMA(21) for major bias filter (prevents counter-trend trades)
3. Fisher Transform(9) for entry timing (better than RSI for reversals in bear/range)
4. ATR volatility filter (avoid extreme vol spikes that cause whipsaws)
5. NO session filter (killed trade frequency in exp #010, #015)
6. NO volume filter (too restrictive for 1h timeframe)
7. 2.5x ATR trailing stoploss

Why Fisher Transform over RSI:
- Fisher normalizes price to Gaussian distribution, better for spotting extremes
- Crosses at -1.5/+1.5 are clearer reversal signals than RSI 30/70
- Works better in bear/range markets (2022 crash, 2025 test period)

Position sizing: 0.20-0.30 discrete (smaller for 1h vs 4h/12h)
Target trades: 40-80/year (1h needs more than 12h but less than 30m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h12h_hma_trend_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Better for spotting reversal extremes than RSI in bear/range markets.
    """
    # Calculate typical price
    typical = (high + low + close) / 2.0
    typical_s = pd.Series(typical)
    
    # Normalize to -1 to +1 range
    highest = typical_s.rolling(window=period, min_periods=period).max()
    lowest = typical_s.rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = np.where(range_val == 0, 0.001, range_val)
    
    normalized = 0.66 * ((typical - lowest) / range_val - 0.5) + 0.67 * np.roll(normalized, 1) if len(normalized := 0.66 * ((typical - lowest) / range_val - 0.5)) > 0 else np.zeros(len(typical))
    
    # Recalculate properly
    normalized = np.zeros(len(typical))
    for i in range(period, len(typical)):
        if range_val[i] > 0.001:
            normalized[i] = 0.66 * ((typical[i] - lowest.iloc[i]) / range_val.iloc[i] - 0.5) + 0.67 * normalized[i-1]
        else:
            normalized[i] = normalized[i-1] if i > 0 else 0.0
    
    # Clamp to avoid extreme values
    normalized = np.clip(normalized, -0.99, 0.99)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 0.001))
    fisher = pd.Series(fisher).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return fisher

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4H indicators
    hma_4h_16 = calculate_hma(df_4h['close'].values, 16)
    hma_4h_48 = calculate_hma(df_4h['close'].values, 48)
    
    # Calculate 12H indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher_transform(high, low, close, 9)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    # Smaller for 1h vs 4h/12h to account for more noise
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            continue
        
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]):
            continue
        
        # === 4H HMA TREND ===
        hma_4h_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === 12H BIAS ===
        bias_bullish = close[i] > hma_12h_21_aligned[i]
        bias_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === VOLATILITY FILTER ===
        # Avoid trading during extreme vol spikes (whipsaw risk)
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_normal = 0.4 < atr_ratio < 2.5
        else:
            vol_normal = True
            atr_ratio = 1.0
        
        # === POSITION SIZING ===
        # Reduce size when volatility is elevated
        vol_adjustment = np.clip(1.0 / atr_ratio if atr_ratio > 0 else 1.0, 0.6, 1.2)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.15, 0.30)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Fisher Transform thresholds for reversals
        fisher_prev = fisher[i-1] if i > 0 else fisher[i]
        
        # LONG: 4h bullish + 12h bias bullish + Fisher reversal from oversold + vol normal
        if hma_4h_bullish and bias_bullish and vol_normal:
            # Fisher crosses above -1.5 from below (oversold reversal)
            if fisher_prev <= -1.5 and fisher[i] > -1.5:
                new_signal = current_size
            # Or Fisher rising from deep oversold with momentum
            elif fisher[i] > -1.0 and fisher_prev < fisher[i] and fisher[i-2] < fisher_prev if i > 1 else False:
                if fisher[i] < 1.0:  # Not yet overbought
                    new_signal = current_size
        
        # SHORT: 4h bearish + 12h bias bearish + Fisher reversal from overbought + vol normal
        elif hma_4h_bearish and bias_bearish and vol_normal:
            # Fisher crosses below +1.5 from above (overbought reversal)
            if fisher_prev >= 1.5 and fisher[i] < 1.5:
                new_signal = -current_size
            # Or Fisher falling from deep overbought with momentum
            elif fisher[i] < 1.0 and fisher_prev > fisher[i] and fisher[i-2] > fisher_prev if i > 1 else False:
                if fisher[i] > -1.0:  # Not yet oversold
                    new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~2.5 days on 1h), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and bias_bullish and fisher[i] > -0.5:
                new_signal = current_size * 0.5
            elif hma_4h_bearish and bias_bearish and fisher[i] < 0.5:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
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
            if position_side > 0 and hma_4h_bearish:
                trend_reversal = True
            if position_side < 0 and hma_4h_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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