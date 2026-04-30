#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
# Uses weekly EMA50 for stable weekly trend direction and requires volume > 1.8x 20-period average.
# Trades only during 08-20 UTC session to avoid low-liquidity periods.
# Designed for low trade frequency (~10-25 trades/year) to minimize fee drag on daily timeframe.
# Camarilla levels provide strong support/resistance that work in both trending and ranging markets.
# Added ATR-based stoploss to limit drawdowns during adverse moves.
# Focus on BTC/ETH as primary targets with SOL as secondary.

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_VolumeConfirm_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA50 and ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (1.8 * vol_ma_20)
        else:
            volume_confirm = False
        
        # Calculate Camarilla levels for current day using previous day's OHLC
        # Need to group by day to get previous day's OHLC
        if i >= 24:  # Need at least 24 hours of data for previous day
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
                        camarilla_r3 = prev_close + (range_val * 1.1 / 4)
                        camarilla_s3 = prev_close - (range_val * 1.1 / 4)
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
            # Long: price breaks above Camarilla R3, 1w EMA50 uptrend, volume spike
            if (curr_close > camarilla_r3 and 
                curr_close > curr_ema_50_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla S3, 1w EMA50 downtrend, volume spike
            elif (curr_close < camarilla_s3 and 
                  curr_close < curr_ema_50_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Camarilla S3 OR stoploss hit
            if (curr_close < camarilla_s3 or 
                curr_close < entry_price - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Camarilla R3 OR stoploss hit
            if (curr_close > camarilla_r3 or 
                curr_close > entry_price + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals