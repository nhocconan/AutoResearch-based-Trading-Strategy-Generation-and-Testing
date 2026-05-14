# Strategy: 12h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.096 | +18.9% | -5.9% | 252 | FAIL |
| ETHUSDT | 0.046 | +22.4% | -7.9% | 201 | PASS |
| SOLUSDT | 0.857 | +86.8% | -13.1% | 203 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.341 | +8.9% | -3.8% | 92 | PASS |
| SOLUSDT | 0.418 | +9.6% | -4.5% | 54 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 with 1d EMA34 uptrend and volume > 2.0x 20-bar average.
# Short when price breaks below S3 with 1d EMA34 downtrend and volume confirmation.
# Uses discrete sizing 0.30. ATR-based stoploss (signal→0 when price moves against position by 2.5*ATR).
# Primary timeframe: 12h, HTF: 1d for EMA trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.
# Session filter: 08-20 UTC to reduce noise trades.

name = "12h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels (R3, S3) from previous day's OHLC
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.zeros_like(daily_close)
    camarilla_s3 = np.zeros_like(daily_close)
    
    for i in range(len(daily_close)):
        if i == 0:
            camarilla_r3[i] = daily_close[i]  # fallback for first day
            camarilla_s3[i] = daily_close[i]
        else:
            prev_close = daily_close[i-1]
            prev_high = daily_high[i-1]
            prev_low = daily_low[i-1]
            camarilla_r3[i] = prev_close + 1.1 * (prev_high - prev_low) / 4
            camarilla_s3[i] = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 34  # warmup for EMA34 and ATR
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-bar average (tight to reduce trades)
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 2.0)
        
        # Camarilla breakout conditions
        breakout_long = curr_high > camarilla_r3_aligned[i]  # price breaks above R3
        breakout_short = curr_low < camarilla_s3_aligned[i]  # price breaks below S3
        
        # Trend filter: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = curr_close > ema_34_aligned[i]
        bearish_trend = curr_close < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 AND bullish trend AND volume confirmation
            if (breakout_long and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short: Breakout below S3 AND bearish trend AND volume confirmation
            elif (breakout_short and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below S3 OR trend turns bearish
            elif (curr_low < camarilla_s3_aligned[i] or 
                  bearish_trend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above R3 OR trend turns bullish
            elif (curr_high > camarilla_r3_aligned[i] or 
                  bullish_trend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-01 07:43
