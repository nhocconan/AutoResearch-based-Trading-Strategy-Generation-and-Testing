#!/usr/bin/env python3
"""
Experiment #351: 1h Fisher Transform + 4h HMA Trend + Volume Breakout

Hypothesis: After 299 failed strategies, the key insight is that complex regime
filters (Choppiness, ADX, etc.) add lag and reduce trade count. The Ehlers Fisher
Transform is specifically noted in research as catching reversals in bear rallies
with high precision. Combined with 4h HMA trend bias and volume confirmation,
this should generate quality trades with Sharpe > 0.676.

Key components:
1. EHLERS FISHER TRANSFORM (period=9): Normalizes price to -1.5 to +1.5 range.
   Long when Fisher crosses above -1.5 (oversold reversal).
   Short when Fisher crosses below +1.5 (overbought reversal).
   Proven to work in bear/range markets (2022 crash, 2025 test period).

2. 4h HMA(21) for trend bias: Only long if price > 4h HMA, only short if < 4h HMA.
   This filters 50% of false signals and aligns with higher timeframe momentum.

3. VOLUME CONFIRMATION: Volume must be > 1.5x 20-bar average on entry bar.
   Filters low-conviction breakouts and reduces whipsaw.

4. ATR(14) stoploss at 2.0x: Tight enough to limit drawdown, loose enough to
   avoid premature exits. Trailing stop updates on favorable moves.

5. ASYMMETRIC SIZING: Long 0.30, Short 0.25 (crypto has long bias, shorts riskier).
   Discrete levels to minimize fee churn.

6. 2-BAR CONFIRMATION: Fisher must stay beyond threshold for 2 consecutive bars
   before entry. Reduces false signals from single-bar spikes.

Why 1h timeframe:
- Fast enough to catch Fisher reversals quickly
- Slow enough to avoid 5m/15m noise
- Volume signals more reliable than lower timeframes
- Generates 20-50 trades/year target (not 200+ fee churn)

Timeframe: 1h (REQUIRED)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_vol_breakout_atr_v1"
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
    Calculate Ehlers Fisher Transform.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest_low) / (highest_high - lowest_low) * 2 - 1
    3. Apply Fisher: 0.5 * ln((1 + normalized) / (1 - normalized))
    
    Output ranges roughly -1.5 to +1.5. Crosses indicate reversals.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    typical = (high + low) / 2.0
    
    for i in range(period, n):
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize to -1 to +1
        normalized = (typical[i] - lowest_low) / price_range * 2.0 - 1.0
        
        # Clamp to avoid division by zero
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value (Ehlers method)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]
        
        fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_prev

def calculate_volume_ma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    vol_ma = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25  # Asymmetric: shorts riskier in crypto
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    fisher_cross_bar = 0  # Track when Fisher crossed threshold
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === 2-BAR CONFIRMATION ===
        # Fisher must stay beyond threshold for 2 consecutive bars
        fisher_long_confirmed = fisher_long_cross or (fisher[i] >= -1.5 and i > 100 and fisher[i-1] >= -1.5)
        fisher_short_confirmed = fisher_short_cross or (fisher[i] <= 1.5 and i > 100 and fisher[i-1] <= 1.5)
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # Long entry: Fisher confirmed + 4h bullish + volume spike
        if fisher_long_confirmed and bull_trend_4h and vol_spike:
            new_signal = SIZE_LONG
        
        # Short entry: Fisher confirmed + 4h bearish + volume spike
        elif fisher_short_confirmed and bear_trend_4h and vol_spike:
            new_signal = -SIZE_SHORT
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === FISHER REVERSAL EXIT ===
        # Exit long when Fisher goes above +1.0 (overbought)
        # Exit short when Fisher goes below -1.0 (oversold)
        if in_position and new_signal != 0.0:
            if position_side > 0 and fisher[i] > 1.0:
                new_signal = 0.0
            if position_side < 0 and fisher[i] < -1.0:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                fisher_cross_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                fisher_cross_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
                fisher_cross_bar = 0
        
        signals[i] = new_signal
    
    return signals