# Strategy: 12h_Camarilla_R1S1_1dEMA34_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.116 | +25.0% | -5.1% | 134 | PASS |
| ETHUSDT | 0.096 | +24.4% | -9.9% | 108 | PASS |
| SOLUSDT | 0.385 | +48.1% | -16.9% | 110 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.022 | -2.2% | -7.8% | 49 | FAIL |
| ETHUSDT | 0.477 | +12.2% | -5.1% | 40 | PASS |
| SOLUSDT | -0.761 | -3.4% | -10.9% | 42 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R1 with 1d EMA34 uptrend and volume > 1.8x 20-bar average.
# Short when price breaks below S1 with 1d EMA34 downtrend and volume confirmation.
# Uses discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
# Primary timeframe: 12h, HTF: 1d for EMA trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Camarilla_R1S1_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    
    # Calculate Camarilla levels (R1, S1) from previous day's OHLC
    df_1d_cama = get_htf_data(prices, '1d')
    if len(df_1d_cama) < 1:
        return np.zeros(n)
    
    # Extract daily OHLC values
    daily_open = df_1d_cama['open'].values
    daily_high = df_1d_cama['high'].values
    daily_low = df_1d_cama['low'].values
    daily_close = df_1d_cama['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r1 = np.zeros_like(daily_close)
    camarilla_s1 = np.zeros_like(daily_close)
    
    for i in range(len(daily_close)):
        if i == 0:
            camarilla_r1[i] = daily_close[i]  # fallback for first day
            camarilla_s1[i] = daily_close[i]
        else:
            prev_close = daily_close[i-1]
            prev_high = daily_high[i-1]
            prev_low = daily_low[i-1]
            camarilla_r1[i] = prev_close + 1.1 * (prev_high - prev_low) / 12
            camarilla_s1[i] = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d_cama, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d_cama, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 34  # warmup for EMA34 and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-bar average (balanced to reduce trades)
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.8)
        
        # Camarilla breakout conditions
        breakout_long = curr_high > camarilla_r1_aligned[i]  # price breaks above R1
        breakout_short = curr_low < camarilla_s1_aligned[i]  # price breaks below S1
        
        # Trend filter: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = curr_close > ema_34_aligned[i]
        bearish_trend = curr_close < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R1 AND bullish trend AND volume confirmation
            if (breakout_long and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Breakout below S1 AND bearish trend AND volume confirmation
            elif (breakout_short and 
                  bearish_trend and 
                  volume_confirm):
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
            # Exit: price breaks below S1 OR trend turns bearish
            elif (curr_low < camarilla_s1_aligned[i] or 
                  bearish_trend):
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
            # Exit: price breaks above R1 OR trend turns bullish
            elif (curr_high > camarilla_r1_aligned[i] or 
                  bullish_trend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 07:34
