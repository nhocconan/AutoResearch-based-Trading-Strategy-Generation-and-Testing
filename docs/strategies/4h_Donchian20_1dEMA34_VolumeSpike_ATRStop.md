# Strategy: 4h_Donchian20_1dEMA34_VolumeSpike_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.114 | +25.3% | -14.9% | 73 | PASS |
| ETHUSDT | 0.173 | +29.6% | -15.7% | 80 | PASS |
| SOLUSDT | 1.034 | +212.2% | -25.7% | 69 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.749 | -3.7% | -9.2% | 34 | FAIL |
| ETHUSDT | 0.957 | +26.9% | -7.2% | 20 | PASS |
| SOLUSDT | 0.365 | +13.0% | -12.2% | 20 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike confirmation, and ATR(14) stoploss.
# Long when price breaks above Donchian(20) high + volume spike in bull trend (close > 1d EMA34).
# Short when price breaks below Donchian(20) low + volume spike in bear trend (close < 1d EMA34).
# Uses discrete position sizing (0.25) to minimize fee churn and ATR-based exits for risk control.
# Designed for 75-200 total trades over 4 years (19-50/year) with Sharpe > 1.0 on BTC/ETH/SOL.

name = "4h_Donchian20_1dEMA34_VolumeSpike_ATRStop"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Breakout conditions
        long_breakout = close_val > donch_high[i-1]  # Break above previous period's high
        short_breakout = close_val < donch_low[i-1]  # Break below previous period's low
        
        # Entry logic
        if position == 0:
            if is_bull_trend and long_breakout and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif is_bear_trend and short_breakout and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: stoploss (2*ATR) or trend reversal
            if close_val <= entry_price - 2.0 * atr[i] or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: stoploss (2*ATR) or trend reversal
            if close_val >= entry_price + 2.0 * atr[i] or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 05:59
