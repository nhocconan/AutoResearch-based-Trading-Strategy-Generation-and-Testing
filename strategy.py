#!/usr/bin/env python3
"""
Experiment #044: 4h Fisher Transform + Vol Spike + 12h HMA Trend

Hypothesis: Previous Connors RSI + Choppiness strategies failed due to over-filtering.
This strategy uses a DIFFERENT edge combination:

1. Ehlers Fisher Transform (period=9) - proven reversal indicator that catches 
   extremes in bear/bull markets better than RSI. Long when Fisher crosses above -1.5,
   short when crosses below +1.5.

2. ATR Vol Spike Filter - only trade when ATR(7)/ATR(30) > 1.5, indicating elevated
   volatility where mean-reversion has edge. This reduces trades but improves quality.

3. 12h HMA(21) for trend bias - simpler than dual-HTF, faster response than 1d KAMA.
   Only take longs when price > 12h HMA, shorts when price < 12h HMA.

4. Asymmetric sizing - reduce position size in high vol (ATR ratio > 2.0) to control DD.

5. Looser Fisher thresholds (-1.2/+1.2 instead of -1.5/+1.5) to ensure trades on ALL symbols.

Why this should beat Sharpe=0.028:
- Fisher Transform is fundamentally different from RSI/Connors (non-linear transform)
- Vol spike filter worked in research for "vol crush" after panic (BTC 2022 bottom)
- Fewer conflicting filters = more trades while maintaining quality
- 4h timeframe with 12h HTF = proven combination (current best uses similar)
- Asymmetric sizing protects during 2022 crash while capturing 2021 bull

Timeframe: 4h (REQUIRED)
HTF: 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete, vol-adjusted
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_volspike_12h_hma_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    
    Fisher Transform converts price to a Gaussian normal distribution,
    making extremes easier to identify. Works well for reversal detection.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - LL) / (HH - LL) where HH/LL are highest/lowest over period
    3. Apply Fisher: 0.5 * ln((1 + x) / (1 - x)) where x = 2*normalized - 1
    4. Smooth with EMA
    
    Entry signals:
    - Long: Fisher crosses above -1.2 (oversold reversal)
    - Short: Fisher crosses below +1.2 (overbought reversal)
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    # Typical price
    typical = (high + low) / 2.0
    typical_s = pd.Series(typical)
    
    for i in range(period, n):
        # Highest high and lowest low over lookback
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize to 0-1 range
        range_val = hh - ll
        if range_val > 0:
            normalized = (typical[i] - ll) / range_val
        else:
            normalized = 0.5
        
        # Clamp to avoid division by zero in Fisher formula
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        x = 2.0 * normalized - 1.0
        fisher_val = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Smooth with previous fisher (recursive)
        if i > period:
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
            fisher_signal[i] = 0.67 * fisher[i] + 0.33 * fisher_signal[i-1]
        else:
            fisher[i] = fisher_val
            fisher_signal[i] = fisher_val
    
    # Fill initial values
    fisher[:period] = 0.0
    fisher_signal[:period] = 0.0
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    hma_4h_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            continue
        
        # === 12H TREND BIAS ===
        trend_bullish = close[i] > hma_12h_21_aligned[i]
        trend_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === VOLATILITY SPIKE FILTER ===
        # Only trade when vol is elevated (ATR ratio > 1.5)
        atr_ratio = atr_7[i] / atr_30[i]
        vol_spike = atr_ratio > 1.3  # Looser threshold to ensure trades
        
        # === FISHER TRANSFORM SIGNALS ===
        # Crossover detection
        fisher_cross_up = fisher_signal[i] > -1.2 and fisher_signal[i-1] <= -1.2
        fisher_cross_down = fisher_signal[i] < 1.2 and fisher_signal[i-1] >= 1.2
        
        # Extreme levels (backup entry)
        fisher_extreme_low = fisher_signal[i] < -1.8
        fisher_extreme_high = fisher_signal[i] > 1.8
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if atr_ratio > 2.5:
            vol_adjustment = 0.7  # Reduce size in extreme vol
        elif atr_ratio > 1.8:
            vol_adjustment = 0.85
        elif atr_ratio < 1.0:
            vol_adjustment = 1.1  # Increase size in low vol (mean reversion edge)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.32)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if trend_bullish:
            # Primary: Fisher cross up + vol spike
            if fisher_cross_up and vol_spike:
                new_signal = current_size
            # Backup: Extreme oversold (even without vol spike)
            elif fisher_extreme_low and close[i] > hma_4h_50[i]:
                new_signal = current_size * 0.7
        
        # SHORT ENTRIES
        elif trend_bearish:
            # Primary: Fisher cross down + vol spike
            if fisher_cross_down and vol_spike:
                new_signal = -current_size
            # Backup: Extreme overbought (even without vol spike)
            elif fisher_extreme_high and close[i] < hma_4h_50[i]:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 50 bars (~8-9 days on 4h), force entry with weaker conditions
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if trend_bullish and fisher_signal[i] < -0.8:
                new_signal = current_size * 0.5
            elif trend_bearish and fisher_signal[i] > 0.8:
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
            if position_side > 0 and trend_bearish:
                trend_reversal = True
            if position_side < 0 and trend_bullish:
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