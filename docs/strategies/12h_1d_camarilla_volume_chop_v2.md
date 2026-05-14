# Strategy: 12h_1d_camarilla_volume_chop_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.052 | +16.9% | -18.3% | 175 | FAIL |
| ETHUSDT | 0.284 | +37.9% | -27.5% | 151 | PASS |
| SOLUSDT | 0.535 | +76.8% | -26.9% | 137 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.094 | +6.7% | -11.8% | 50 | PASS |
| SOLUSDT | -0.940 | -12.2% | -29.3% | 45 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Uses 1d Camarilla pivot levels (H3/L3) for breakout entries on 12h timeframe
# - Volume confirmation: 12h volume > 1.5x 20-period average to ensure breakout strength
# - Regime filter: 12h Choppiness Index > 61.8 for mean-reversion (fade extreme touches)
# - ATR(14) trailing stop at 2.5x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Novelty: Combines Camarilla pivots with chop regime to avoid false breakouts in sideways markets
# - Works in both bull/bear: Camarilla levels adapt to volatility, chop filter prevents whipsaws

name = "12h_1d_camarilla_volume_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (H3, L3)
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + (1.1 * (high_1d - low_1d) / 4)
    camarilla_l3 = close_1d - (1.1 * (high_1d - low_1d) / 4)
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume > 1.5x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    # 12h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,n) - min(low,n))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    # Avoid division by zero
    chop = np.where(range_14 > 0, 100 * np.log10(atr_14 / range_14) / np.log10(14), 50)
    chop = np.where(np.isnan(chop), 50, chop)
    chop_regime = chop > 61.8  # True when ranging/markets suitable for mean reversion
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop_regime[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.5x ATR from high OR Camarilla L3 touch (mean reversion in chop)
            if low[i] <= highest_since_entry - (2.5 * atr[i]) or \
               (chop_regime[i] and low[i] <= camarilla_l3_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low OR Camarilla H3 touch (mean reversion in chop)
            if high[i] >= lowest_since_entry + (2.5 * atr[i]) or \
               (chop_regime[i] and high[i] >= camarilla_h3_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation
            # Long: price breaks above Camarilla H3 AND volume spike
            if high[i] >= camarilla_h3_aligned[i] and volume_spike[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price breaks below Camarilla L3 AND volume spike
            elif low[i] <= camarilla_l3_aligned[i] and volume_spike[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 22:43
