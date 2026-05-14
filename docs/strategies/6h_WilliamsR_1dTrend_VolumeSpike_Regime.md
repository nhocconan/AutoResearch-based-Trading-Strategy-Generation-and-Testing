# Strategy: 6h_WilliamsR_1dTrend_VolumeSpike_Regime

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.676 | +6.7% | -13.1% | 58 | FAIL |
| ETHUSDT | 0.429 | +35.8% | -8.6% | 46 | PASS |
| SOLUSDT | 0.828 | +64.2% | -6.2% | 33 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.198 | +8.0% | -5.7% | 20 | PASS |
| SOLUSDT | -1.878 | -4.5% | -6.6% | 11 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume spike confirmation.
# Williams %R measures overbought/oversold levels. In bull regime (price > 1d EMA34),
# we go long when Williams %R crosses above -50 from below (bullish momentum).
# In bear regime (price < 1d EMA34), we go short when Williams %R crosses below -50 from above (bearish momentum).
# Volume spike confirms institutional participation. This adapts to both bull and bear markets
# by using the higher timeframe trend for regime filtering and Williams %R for precise entry timing.

name = "6h_WilliamsR_1dTrend_VolumeSpike_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 14-period Williams %R (primary timeframe)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        wr = williams_r[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if np.isnan(wr) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA34, bear if close < 1d EMA34
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: Williams %R crosses above -50 from below (bullish momentum) with volume spike
            long_entry = (wr > -50) and (williams_r[i-1] <= -50) and vol_spike if i > 0 else False
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Williams %R crosses below -50 from above (bearish momentum) with volume spike
            short_entry = (wr < -50) and (williams_r[i-1] >= -50) and vol_spike if i > 0 else False
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on Williams %R crossing below -50 (loss of bullish momentum) or regime change to bear
            if wr < -50 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Williams %R crossing above -50 (loss of bearish momentum) or regime change to bull
            if wr > -50 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 05:06
