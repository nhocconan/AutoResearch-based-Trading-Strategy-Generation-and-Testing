# Strategy: 6h_WilliamsFractal_Breakout_1dEMA34_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.343 | +35.8% | -11.6% | 40 | PASS |
| ETHUSDT | 0.507 | +48.5% | -11.3% | 35 | PASS |
| SOLUSDT | 1.171 | +180.4% | -16.1% | 40 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.960 | -0.5% | -5.7% | 17 | FAIL |
| ETHUSDT | 0.392 | +10.6% | -6.6% | 12 | PASS |
| SOLUSDT | -0.208 | +2.9% | -7.1% | 12 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation
# Long when price breaks above recent bearish fractal AND 1d close > 1d EMA34 AND volume > 2x 20-period average
# Short when price breaks below recent bullish fractal AND 1d close < 1d EMA34 AND volume > 2x 20-period average
# Exit when price crosses 1d EMA34 (trend reversal)
# Uses 6h primary timeframe with 1d HTF for trend filter and fractal structure
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-150 total trades over 4 years (19-38/year) to avoid fee drag
# Williams Fractals identify key swing points; 1d EMA34 filters for higher-timeframe trend; volume confirms participation
# Works in bull markets via breakouts and in bear markets via trend-filtered shorts

name = "6h_WilliamsFractal_Breakout_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for trend filter and fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Fractals on 1d data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] < high[n-1] > high[n+1] > high[n+2]
    # Simplified: high[n-2] < high[n-1] and high[n] < high[n-1] and high[n-3] < high[n-2] and high[n+1] < high[n-1] and high[n+2] < high[n-1]
    # We'll use the mtf_data helper if available, otherwise implement manually
    try:
        from mtf_data import compute_williams_fractals
        bearish_fractal, bullish_fractal = compute_williams_fractals(
            df_1d['high'].values,
            df_1d['low'].values,
        )
        # Align fractals with 2-bar delay for confirmation (needs 2 future 1d bars to confirm)
        bearish_fractal_aligned = align_htf_to_ltf(
            prices, df_1d, bearish_fractal, additional_delay_bars=2
        )
        bullish_fractal_aligned = align_htf_to_ltf(
            prices, df_1d, bullish_fractal, additional_delay_bars=2
        )
    except ImportError:
        # Manual implementation if helper not available
        n_1d = len(df_1d)
        bearish_fractal = np.full(n_1d, np.nan)
        bullish_fractal = np.full(n_1d, np.nan)
        
        for i in range(2, n_1d - 2):
            # Bearish fractal: highest high in the middle
            if (df_1d['high'].iloc[i-2] < df_1d['high'].iloc[i-1] and
                df_1d['high'].iloc[i] < df_1d['high'].iloc[i-1] and
                df_1d['high'].iloc[i-3] < df_1d['high'].iloc[i-2] and
                df_1d['high'].iloc[i+1] < df_1d['high'].iloc[i-1] and
                df_1d['high'].iloc[i+2] < df_1d['high'].iloc[i-1]):
                bearish_fractal[i] = df_1d['high'].iloc[i-1]
            
            # Bullish fractal: lowest low in the middle
            if (df_1d['low'].iloc[i-2] > df_1d['low'].iloc[i-1] and
                df_1d['low'].iloc[i] > df_1d['low'].iloc[i-1] and
                df_1d['low'].iloc[i-3] > df_1d['low'].iloc[i-2] and
                df_1d['low'].iloc[i+1] > df_1d['low'].iloc[i-1] and
                df_1d['low'].iloc[i+2] > df_1d['low'].iloc[i-1]):
                bullish_fractal[i] = df_1d['low'].iloc[i-1]
        
        # Align fractals with 2-bar delay for confirmation
        bearish_fractal_aligned = align_htf_to_ltf(
            prices, df_1d, bearish_fractal, additional_delay_bars=2
        )
        bullish_fractal_aligned = align_htf_to_ltf(
            prices, df_1d, bullish_fractal, additional_delay_bars=2
        )
    
    # Volume confirmation: volume > 2x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above recent bearish fractal AND 1d close > 1d EMA34 AND volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below recent bullish fractal AND 1d close < 1d EMA34 AND volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal)
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal)
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 05:06
