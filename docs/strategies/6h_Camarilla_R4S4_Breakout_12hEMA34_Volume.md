# Strategy: 6h_Camarilla_R4S4_Breakout_12hEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.381 | +40.7% | -10.7% | 77 | PASS |
| ETHUSDT | 0.028 | +19.5% | -18.4% | 70 | PASS |
| SOLUSDT | 0.478 | +66.5% | -25.6% | 56 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.523 | +0.7% | -6.8% | 23 | FAIL |
| ETHUSDT | 0.295 | +10.2% | -9.0% | 24 | PASS |
| SOLUSDT | -0.248 | +0.6% | -13.1% | 19 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 12h EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R4 (strong bullish breakout level) in bull trend (close > 12h EMA34) with volume > 2x 20-period MA.
# Short when price breaks below S4 (strong bearish breakdown level) in bear trend (close < 12h EMA34) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Camarilla R4/S4 levels represent stronger institutional barriers than R3/S3, reducing false breakouts.
# 12h EMA34 provides intermediate-term trend filter to avoid counter-trend trades.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on BTC/ETH.

name = "6h_Camarilla_R4S4_Breakout_12hEMA34_Volume"
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
    
    # Get 12h data for EMA34 and 1d data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 34 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from prior 1d candle (HLC)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_val = df_1d['high'] - df_1d['low']
    
    camarilla_pivot = typical_price.values
    # Camarilla R4 = CP + (H-L)*1.1, S4 = CP - (H-L)*1.1
    camarilla_r4 = camarilla_pivot + (range_val * 1.1)
    camarilla_s4 = camarilla_pivot - (range_val * 1.1)
    
    # Align Camarilla levels to 6h timeframe (use prior completed 1d bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_12h_aligned[i]
        r4_level = camarilla_r4_aligned[i]
        s4_level = camarilla_s4_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Breakout conditions at R4/S4 (stronger levels)
        breakout_long = close_val > r4_level
        breakout_short = close_val < s4_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_long and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and breakout_short and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S4 OR trend reversal
            if close_val < s4_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R4 OR trend reversal
            if close_val > r4_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 06:18
