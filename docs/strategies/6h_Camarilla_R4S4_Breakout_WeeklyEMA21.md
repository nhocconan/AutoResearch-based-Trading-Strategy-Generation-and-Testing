# Strategy: 6h_Camarilla_R4S4_Breakout_WeeklyEMA21

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.499 | +11.3% | -5.0% | 267 | FAIL |
| ETHUSDT | 0.522 | +35.9% | -3.5% | 241 | PASS |
| SOLUSDT | 0.872 | +66.7% | -8.0% | 257 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.066 | +6.5% | -3.7% | 69 | PASS |
| SOLUSDT | -0.628 | +2.3% | -4.5% | 73 | FAIL |

## Code
```python
# 1. Hypothesis: 6h timeframe strategy using Fibonacci-based pivot points (Camarilla) from daily timeframe for breakout entries, filtered by weekly trend (EMA21) and volume confirmation.  
# The strategy aims to capture breakouts in both bull and bear markets by leveraging the statistical tendency of price to revert to or break through key pivot levels (R4/S4) derived from the prior day's range.  
# Weekly EMA21 ensures alignment with the higher timeframe trend, reducing counter-trend trades. Volume confirmation increases the likelihood of sustained moves.  
# Designed for 6h timeframe to balance trade frequency (target: 50-150 total trades over 4 years) and signal reliability, avoiding excessive churn from lower timeframes.  
# Uses discrete position sizing (0.25) to minimize fee churn while maintaining meaningful exposure.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily high/low/close for calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range for pivot calculations
    daily_range = high_1d - low_1d
    
    # Calculate weekly data once for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly range for pivot calculations (though not used, kept for structure)
    weekly_range = high_1w - low_1w
    
    # Camarilla pivot levels (based on previous day)
    camarilla_r4 = close_1d + daily_range * 1.1 / 2
    camarilla_s4 = close_1d - daily_range * 1.1 / 2
    
    # Weekly EMA21 for trend
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align Camarilla levels and weekly EMA to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Weekly trend filter: price above/below weekly EMA21
        price_above_weekly_ema = close[i] > ema_21_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_21_1w_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above R4 with volume and weekly uptrend
        # Short: price breaks below S4 with volume and weekly downtrend
        long_entry = (close[i] > r4_aligned[i]) and price_above_weekly_ema and vol_filter
        short_entry = (close[i] < s4_aligned[i]) and price_below_weekly_ema and vol_filter
        
        # Exit conditions: price returns to opposite S4/R4 levels or weekly trend reversal
        long_exit = (close[i] < s4_aligned[i]) or (not price_above_weekly_ema)
        short_exit = (close[i] > r4_aligned[i]) or (not price_below_weekly_ema)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_WeeklyEMA21"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-28 06:59
