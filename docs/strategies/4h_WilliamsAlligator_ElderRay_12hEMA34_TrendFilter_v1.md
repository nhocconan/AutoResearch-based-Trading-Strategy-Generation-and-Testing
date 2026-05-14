# Strategy: 4h_WilliamsAlligator_ElderRay_12hEMA34_TrendFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.067 | +13.0% | -18.0% | 280 | FAIL |
| ETHUSDT | 0.153 | +27.8% | -18.0% | 294 | PASS |
| SOLUSDT | 0.810 | +158.2% | -28.9% | 301 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.059 | +5.8% | -11.4% | 97 | PASS |
| SOLUSDT | 0.021 | +4.5% | -16.1% | 88 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray + 12h EMA trend filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for trend filter (price above/below EMA34).
- Entry: Long when Alligator is bullish (jaw < teeth < lips) AND Elder Ray bull power > 0 AND price > 12h EMA34.
         Short when Alligator is bearish (jaw > teeth > lips) AND Elder Ray bear power < 0 AND price < 12h EMA34.
- Exit: Opposite Alligator alignment OR price crosses 12h EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Alligator identifies trend absence/presence and direction via smoothed medians.
- Elder Ray measures bull/bear power behind the move.
- Works in bull markets (buy when all bullish aligned) and bear markets (sell when all bearish aligned).
- Estimated trades: ~120 total over 4 years (~30/year) based on Alligator alignment frequency with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def smma(values, period):
    """Calculate Smoothed Moving Average (used in Alligator)."""
    # SMMA is similar to EMA but with different smoothing
    return pd.Series(values).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h, additional_delay_bars=1)
    
    # Williams Alligator on 4h (jaw=13, teeth=8, lips=5 SMMA of median price)
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)   # Red line
    lips = smma(median_price, 5)    # Green line
    
    # Elder Ray on 4h (bull power = high - EMA13, bear power = low - EMA13)
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Alligator alignment conditions
    alligator_bullish = (jaw < teeth) & (teeth < lips)  # Jaw < Teeth < Lips
    alligator_bearish = (jaw > teeth) & (teeth > lips)  # Jaw > Teeth > Lips
    
    # Elder Ray conditions
    elder_bullish = bull_power > 0
    elder_bearish = bear_power < 0
    
    # 12h trend filter
    trend_bullish = close > ema34_12h_aligned
    trend_bearish = close < ema34_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for Alligator/EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Alligator alignment OR price crosses 12h EMA34 in opposite direction
        if position != 0:
            # Exit long: Alligator turns bearish OR price falls below 12h EMA34
            if position == 1:
                if alligator_bearish[i] or curr_close < ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Alligator turns bullish OR price rises above 12h EMA34
            elif position == -1:
                if alligator_bullish[i] or curr_close > ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: All aligned in same direction
        if position == 0:
            # Long: Alligator bullish AND Elder Ray bullish AND bullish 12h trend
            if alligator_bullish[i] and elder_bullish[i] and trend_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Elder Ray bearish AND bearish 12h trend
            elif alligator_bearish[i] and elder_bearish[i] and trend_bearish[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_12hEMA34_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 19:45
