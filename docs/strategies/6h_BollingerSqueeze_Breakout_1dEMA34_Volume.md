# Strategy: 6h_BollingerSqueeze_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.114 | +25.3% | -15.1% | 117 | PASS |
| ETHUSDT | 0.291 | +36.1% | -9.5% | 119 | PASS |
| SOLUSDT | 0.495 | +65.1% | -24.5% | 103 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.213 | -4.7% | -6.7% | 39 | FAIL |
| ETHUSDT | 1.149 | +25.7% | -5.4% | 35 | PASS |
| SOLUSDT | 0.046 | +5.9% | -9.3% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Bollinger Band width < 20th percentile indicates low volatility squeeze
# Breakout occurs when price closes outside bands with volume > 1.5x average
# Trend filter: 1d EMA34 - only trade in direction of higher timeframe trend
# Works in both bull/bear markets by capturing volatility expansion after consolidation
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_BollingerSqueeze_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper_band - lower_band) / basis
    # Squeeze: BB width below 20th percentile (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=30).quantile(0.20).values
    squeeze = bb_width < bb_width_percentile
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50  # Need enough data for BB width percentile
    
    for i in range(start_idx, n):
        if (np.isnan(basis[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bollinger Band breakout above upper band with bullish trend and volume spike
            if (close[i] > upper_band[i] and 
                close[i-1] <= upper_band[i-1] and  # Just broke above
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bollinger Band breakout below lower band with bearish trend and volume spike
            elif (close[i] < lower_band[i] and 
                  close[i-1] >= lower_band[i-1] and  # Just broke below
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price closes below basis (mean reversion) OR trend turns bearish
            if close[i] < basis[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price closes above basis (mean reversion) OR trend turns bullish
            if close[i] > basis[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 05:00
