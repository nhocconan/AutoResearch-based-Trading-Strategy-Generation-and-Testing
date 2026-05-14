# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.376 | +39.9% | -9.3% | 248 | KEEP |
| ETHUSDT | 0.107 | +24.8% | -14.2% | 243 | KEEP |
| SOLUSDT | 0.737 | +106.6% | -17.8% | 229 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.547 | -0.1% | -7.2% | 97 | DISCARD |
| ETHUSDT | 0.225 | +9.1% | -12.1% | 85 | KEEP |
| SOLUSDT | 0.403 | +12.3% | -8.2% | 84 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation spike.
# Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND volume > 1.8x 4h volume median.
# Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND volume > 1.8x 4h volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Camarilla levels from prior 1d provide structure; 1d EMA34 filters longer-term trend (more stable than 12h).
# Volume confirmation ensures momentum. Target: 20-35 trades/year on 4h timeframe.
# Proven pattern: tight entries + volume + trend filter works on BTC/ETH in both bull/bear.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume median (20-period for stability)
    vol_median_4h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 1d EMA34 trend (more stable than 12h EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (use same df_1d for consistency)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla: based on prior day's high, low, close
    h1 = df_1d['high'].shift(1).values  # prior day high
    l1 = df_1d['low'].shift(1).values   # prior day low
    c1 = df_1d['close'].shift(1).values # prior day close
    
    # Calculate Camarilla R3 and S3 levels
    # R3 = c1 + (h1 - l1) * 1.1/4
    # S3 = c1 - (h1 - l1) * 1.1/4
    camarilla_range = h1 - l1
    r3 = c1 + camarilla_range * 1.1 / 4.0
    s3 = c1 - camarilla_range * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, volume, and Camarilla
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_median_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price vs 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8x 4h volume median (slightly looser for more trades)
        if vol_median_4h[i] <= 0 or np.isnan(vol_median_4h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_4h[i] * 1.8)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > R3 AND uptrend AND volume spike
            if curr_close > r3_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < S3 AND downtrend AND volume spike
            elif curr_close < s3_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below S3 OR trend turns down
            elif curr_close < s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above R3 OR trend turns up
            elif curr_close > r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 12:02
