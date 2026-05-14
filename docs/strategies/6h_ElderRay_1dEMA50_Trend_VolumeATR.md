# Strategy: 6h_ElderRay_1dEMA50_Trend_VolumeATR

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.131 | +25.9% | -8.1% | 103 | KEEP |
| ETHUSDT | 0.216 | +31.4% | -8.8% | 95 | KEEP |
| SOLUSDT | 1.204 | +188.3% | -20.0% | 82 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.597 | +0.5% | -5.1% | 36 | DISCARD |
| ETHUSDT | 0.064 | +6.3% | -7.7% | 29 | KEEP |
| SOLUSDT | 0.263 | +9.5% | -10.4% | 23 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and ATR-based volume confirmation
# Uses 6h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Elder Ray measures bull/bear power via EMA13 deviation: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d EMA50 provides intermediate-term trend filter to avoid counter-trend entries
# Volume confirmation uses ATR ratio to detect institutional participation during volatility expansion
# Works in bull (continuation via trend filter + bull power) and bear (continuation via trend filter + bear power)
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure
# Designed to avoid overtrading by requiring confluence of trend, momentum, and volume

name = "6h_ElderRay_1dEMA50_Trend_VolumeATR"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13      # Bull Power: High - EMA13
    bear_power = low - ema_13       # Bear Power: Low - EMA13
    
    # Volume confirmation using ATR ratio (6h)
    # True Range calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Volume spike: current volume > 1.5 * 20-period volume EMA during ATR expansion
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema_20
    atr_ratio = atr / pd.Series(atr).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = (vol_ratio > 1.5) & (atr_ratio > 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Bull Power > 0 (bulls in control) with volume confirmation
                if bull_power[i] > 0 and volume_confirmation[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Bear Power < 0 (bears in control) with volume confirmation
                if bear_power[i] < 0 and volume_confirmation[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around 1d EMA50
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 (bulls lose control) or price below 1d EMA50
            if bull_power[i] <= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 (bears lose control) or price above 1d EMA50
            if bear_power[i] >= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 02:09
