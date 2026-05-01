#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and ATR-based trailing stop.
# Uses 1w EMA50 for trend alignment (HTF direction) and ATR(14) for dynamic stoploss.
# Long when price breaks above R3 and above 1w EMA50; short when breaks below S3 and below 1w EMA50.
# Exit on opposite Camarilla level break or ATR trailing stop (2.0x ATR from extreme).
# Session filter (08-20 UTC) reduces noise. Discrete sizing 0.25 minimizes fee churn.
# Target: 12-37 trades/year by using 1w for signal direction and 12h only for entry timing.
# This strategy focuses on BTC/ETH with proven Camarilla structure and weekly trend filter.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_ATRTrail_Session_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w data
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for 12h timeframe trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    start_idx = 100  # warmup for EMA, ATR
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Camarilla levels for current day using previous day's OHLC
        if i >= 2:  # Need at least 2 bars (1 day) of 12h data for previous day
            # Get timestamp of current bar
            curr_time = prices.iloc[i]["open_time"]
            # Get start of current day (00:00 UTC)
            curr_day_start = curr_time.replace(hour=0, minute=0, second=0, microsecond=0)
            # Get start of previous day
            prev_day_start = curr_day_start - pd.Timedelta(days=1)
            # Get end of previous day (23:59:59.999 UTC)
            prev_day_end = curr_day_start - pd.Timedelta(microseconds=1)
            
            # Filter prices for previous day
            mask = (prices["open_time"] >= prev_day_start) & (prices["open_time"] <= prev_day_end)
            if mask.any():
                prev_day_data = prices.loc[mask]
                if len(prev_day_data) > 0:
                    prev_high = prev_day_data["high"].max()
                    prev_low = prev_day_data["low"].min()
                    prev_close = prev_day_data["close"].iloc[-1]
                    
                    # Calculate Camarilla levels
                    range_val = prev_high - prev_low
                    if range_val > 0:
                        camarilla_r3 = prev_close + (range_val * 1.1 / 4)  # R3 level
                        camarilla_s3 = prev_close - (range_val * 1.1 / 4)  # S3 level
                    else:
                        camarilla_r3 = curr_close
                        camarilla_s3 = curr_close
                else:
                    camarilla_r3 = curr_close
                    camarilla_s3 = curr_close
            else:
                camarilla_r3 = curr_close
                camarilla_s3 = curr_close
        else:
            camarilla_r3 = curr_close
            camarilla_s3 = curr_close
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, price above 1w EMA50
            if (curr_close > camarilla_r3 and 
                curr_close > curr_ema):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                long_stop = curr_high - 2.0 * curr_atr  # initial stop below entry
            # Short: price breaks below Camarilla S3, price below 1w EMA50
            elif (curr_close < camarilla_s3 and 
                  curr_close < curr_ema):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                short_stop = curr_low + 2.0 * curr_atr  # initial stop above entry
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update trailing stop: move stop up to highest high minus 2.0*ATR
            long_stop = max(long_stop, curr_high - 2.0 * curr_atr)
            # Exit conditions: price breaks below Camarilla S3 OR stoploss hit
            if (curr_close < camarilla_s3 or 
                curr_close < long_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update trailing stop: move stop down to lowest low plus 2.0*ATR
            short_stop = min(short_stop, curr_low + 2.0 * curr_atr)
            # Exit conditions: price breaks above Camarilla R3 OR stoploss hit
            if (curr_close > camarilla_r3 or 
                curr_close > short_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals