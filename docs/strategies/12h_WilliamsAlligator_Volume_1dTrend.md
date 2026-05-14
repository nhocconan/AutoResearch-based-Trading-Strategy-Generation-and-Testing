# Strategy: 12h_WilliamsAlligator_Volume_1dTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.824 | -3.8% | -18.6% | 274 | DISCARD |
| ETHUSDT | 0.124 | +25.8% | -11.8% | 232 | KEEP |
| SOLUSDT | 0.884 | +117.3% | -26.9% | 191 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.256 | +9.1% | -7.8% | 81 | KEEP |
| SOLUSDT | -0.221 | +2.4% | -8.0% | 75 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with volume confirmation and daily trend filter.
# Uses Alligator (Jaw=13, Teeth=8, Lips=5) to detect trends in higher timeframes.
# Long when Lips > Teeth > Jaw (bullish alignment) with volume > 1.3x average and 1d close > EMA50.
# Short when Lips < Teeth < Jaw (bearish alignment) with volume > 1.3x average and 1d close < EMA50.
# Exit when Alligator lines re-interlace (Trading Zone) or volume drops below average.
# Designed for ~20-30 trades/year with strong trend filtering to avoid whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 1d data
    # Jaw (Blue) = 13-period SMMA, shifted 8 bars forward
    # Teeth (Red) = 8-period SMMA, shifted 5 bars forward
    # Lips (Green) = 5-period SMMA, shifted 3 bars forward
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = np.full_like(series, np.nan)
        if len(series) < period:
            return sma
        sma[period-1] = np.mean(series[:period])
        for i in range(period, len(series)):
            sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Shift forward (Alligator lines are shifted into the future)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.3x 24-period average (24*12h = 12 days)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 24-period volume MA and Alligator lines
    start_idx = max(24, 13)  # Need enough data for Alligator
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter
        vol_filter = vol_now > 1.3 * vol_avg
        
        # Alligator alignment
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Trend filter from 1d EMA50
        bullish_trend = price > ema50_aligned[i]
        bearish_trend = price < ema50_aligned[i]
        
        if position == 0:
            # Long: bullish Alligator alignment with volume and bullish trend
            if bullish_alignment and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: bearish Alligator alignment with volume and bearish trend
            elif bearish_alignment and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator re-interlaces (Trading Zone) or volume drops
            if not bullish_alignment or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Alligator re-interlaces (Trading Zone) or volume drops
            if not bearish_alignment or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_Volume_1dTrend"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-27 10:43
