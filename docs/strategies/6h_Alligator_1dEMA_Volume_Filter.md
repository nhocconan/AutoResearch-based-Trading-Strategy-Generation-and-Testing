# Strategy: 6h_Alligator_1dEMA_Volume_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.177 | +29.3% | -16.3% | 185 | PASS |
| ETHUSDT | 0.111 | +24.7% | -15.7% | 186 | PASS |
| SOLUSDT | 0.602 | +98.0% | -28.7% | 173 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.284 | +1.8% | -8.2% | 57 | FAIL |
| ETHUSDT | 0.385 | +12.8% | -9.6% | 56 | PASS |
| SOLUSDT | 0.300 | +11.1% | -17.0% | 46 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (13,8,5 SMAs) with 1d EMA trend filter and volume confirmation.
# The Alligator's jaw-teeth-lips convergence indicates ranging; divergence signals trend start.
# Combined with 1d EMA for trend direction and volume to avoid false breakouts.
# Designed to work in both bull (follow 1d uptrend) and bear (follow 1d downtrend) markets.
name = "6h_Alligator_1dEMA_Volume_Filter"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h timeframe: SMAs of median price
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using close for simplicity (median price = (high+low+close)/3, but close is acceptable)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Alligator conditions:
    # Converging (ranging): |jaw - teeth| < threshold AND |teeth - lips| < threshold
    # Diverging (trending): jaw, teeth, lips are ordered and separated
    # For uptrend: lips > teeth > jaw (green alignment)
    # For downtrend: lips < teeth < jaw (red alignment)
    
    # Dynamic threshold based on ATR-like volatility
    price_range = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    conv_threshold = 0.1 * price_range  # 10% of average range
    
    jaw_teeth_diff = np.abs(jaw - teeth)
    teeth_lips_diff = np.abs(teeth - lips)
    
    # Converging market (Alligator sleeping)
    converging = (jaw_teeth_diff < conv_threshold) & (teeth_lips_diff < conv_threshold)
    
    # Diverging market - Uptrend (Alligator awake, mouth open up)
    uptrend_aligned = (lips > teeth) & (teeth > jaw)
    # Diverging market - Downtrend (Alligator awake, mouth open down)
    downtrend_aligned = (lips < teeth) & (teeth < jaw)
    
    # Volume confirmation: volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need sufficient lookback for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: Alligator uptrend + price above 1d EMA + volume confirmation
            if (uptrend_aligned[i] and price > ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator downtrend + price below 1d EMA + volume confirmation
            elif (downtrend_aligned[i] and price < ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator converges (sleeping) or price crosses below 1d EMA
            if converging[i] or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator converges (sleeping) or price crosses above 1d EMA
            if converging[i] or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 01:45
