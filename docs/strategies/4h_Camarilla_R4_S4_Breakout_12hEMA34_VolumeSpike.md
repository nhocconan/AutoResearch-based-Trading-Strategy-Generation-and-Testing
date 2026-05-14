# Strategy: 4h_Camarilla_R4_S4_Breakout_12hEMA34_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.180 | +13.2% | -12.7% | 296 | FAIL |
| ETHUSDT | 0.132 | +26.2% | -11.6% | 285 | PASS |
| SOLUSDT | 0.038 | +19.3% | -18.1% | 230 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.450 | +27.6% | -6.2% | 101 | PASS |
| SOLUSDT | 0.520 | +12.8% | -9.0% | 85 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA34 trend filter and volume spike confirmation.
# In bull regime (price > 12h EMA34), go long on breakout above R4 with volume spike.
# In bear regime (price < 12h EMA34), go short on breakdown below S4 with volume spike.
# Uses Camarilla pivot levels from prior 4h for structure, 12h EMA34 for regime filter,
# and 4h volume spike for confirmation. Designed for 75-200 total trades over 4 years.
# Focus on BTC/ETH; SOL as secondary.

name = "4h_Camarilla_R4_S4_Breakout_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots (prior completed 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate prior 4h Camarilla levels (R4, S4)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    range_4h = high_4h - low_4h
    camarilla_r4 = close_4h + 1.1 * range_4h * 1.1 / 2  # R4 level
    camarilla_s4 = close_4h - 1.1 * range_4h * 1.1 / 2  # S4 level
    
    # Align Camarilla levels to 4h (wait for 4h bar to complete)
    r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA34 trend filter
    close_12h = df_12h['close'].values
    ema_34 = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Calculate volume regime: current 4h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(r4) or np.isnan(s4) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 12h EMA34, bear if close < 12h EMA34
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: breakout above R4 with volume spike
            long_entry = (close_val > r4) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: breakdown below S4 with volume spike
            short_entry = (close_val < s4) and vol_spike
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
            # Exit on breakdown below S4 (failure of bullish breakout) or regime change to bear
            if close_val < s4 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on breakout above R4 (failure of bearish breakdown) or regime change to bull
            if close_val > r4 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 05:13
