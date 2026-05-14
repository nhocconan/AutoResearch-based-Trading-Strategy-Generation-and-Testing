# Strategy: 4h_WilliamsFractal_Breakout_12hEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.048 | +22.2% | -15.5% | 106 | PASS |
| ETHUSDT | 0.281 | +35.6% | -10.0% | 106 | PASS |
| SOLUSDT | 1.118 | +184.8% | -27.7% | 116 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.952 | -1.2% | -6.5% | 42 | FAIL |
| ETHUSDT | 0.055 | +6.3% | -10.4% | 44 | PASS |
| SOLUSDT | -0.215 | +2.3% | -9.8% | 34 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Fractal breakout with 12h EMA34 trend filter and volume confirmation.
Long when price breaks above latest bullish fractal AND volume > 1.5x 20-period average AND price > 12h EMA34.
Short when price breaks below latest bearish fractal AND volume > 1.5x 20-period average AND price < 12h EMA34.
Exit when price crosses the 12h EMA34 in opposite direction.
Williams Fractals identify key swing points, 12h EMA34 filters for higher timeframe trend,
volume confirmation reduces false breakouts. Designed to work in both bull and bear markets
by trading with the 12h trend while using fractals for precise entry/exit.
Targets 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for price and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume average (20-period) on 4h
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams Fractals on 4h timeframe (need 5 bars: 2 left, center, 2 right)
    # Using high and low arrays from 4h data
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_4h, low_4h)
    
    # Align all indicators to 4h timeframe
    # Note: Williams fractals need additional 2-bar delay for confirmation
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bearish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators (Williams fractals + EMA34)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_34 = ema_34_12h_aligned[i]
        vol_ma = volume_ma_aligned[i]
        bullish_fract = bullish_fractal_aligned[i]
        bearish_fract = bearish_fractal_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above bullish fractal AND volume > 1.5x avg AND price > 12h EMA34 (bullish trend)
            if high_price > bullish_fract and vol > 1.5 * vol_ma and price > ema_34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish fractal AND volume > 1.5x avg AND price < 12h EMA34 (bearish trend)
            elif low_price < bearish_fract and vol > 1.5 * vol_ma and price < ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 12h EMA34
            if price < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 12h EMA34
            if price > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsFractal_Breakout_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 20:34
