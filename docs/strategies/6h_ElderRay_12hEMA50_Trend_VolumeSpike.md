# Strategy: 6h_ElderRay_12hEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.237 | +32.2% | -9.7% | 178 | PASS |
| ETHUSDT | 0.123 | +25.9% | -15.8% | 163 | PASS |
| SOLUSDT | 1.285 | +240.2% | -24.1% | 128 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.676 | -1.8% | -9.5% | 72 | FAIL |
| ETHUSDT | 0.130 | +7.3% | -9.0% | 58 | PASS |
| SOLUSDT | -0.215 | +0.8% | -14.1% | 53 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Long when Bull Power > 0 + volume spike in bull trend (price > 12h EMA50).
# Short when Bear Power > 0 + volume spike in bear trend (price < 12h EMA50). Uses 12h EMA50 for regime, 6h for entry timing.
# Designed for 50-150 total trades over 4 years. Works in both bull (Bull Power) and bear (Bear Power) regimes.

name = "6h_ElderRay_12hEMA50_Trend_VolumeSpike"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        bp = bull_power[i]
        bb = bear_power[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(bp) or np.isnan(bb):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 12h EMA50, bear if close < 12h EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: Bull Power > 0 (buying strength) + volume spike
            long_entry = (bp > 0) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Bear Power > 0 (selling strength) + volume spike
            short_entry = (bb > 0) and vol_spike
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
            # Exit on Bull Power <= 0 (loss of buying strength) or regime change to bear
            if bp <= 0 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Bear Power <= 0 (loss of selling strength) or regime change to bull
            if bb <= 0 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 05:57
