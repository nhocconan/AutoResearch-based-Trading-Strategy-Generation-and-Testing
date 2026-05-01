#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume confirmation, and chop regime filter.
# Uses 1d EMA34 for strong trend alignment, volume > 1.5x 20-period average for momentum confirmation,
# and Choppiness Index > 61.8 to avoid ranging markets. ATR-based stoploss (2.0x) manages risk.
# Target: 15-30 trades/year by tightening entry conditions to reduce fee drag and improve generalization.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeChopFilter_ATRStop_v1"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d data
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for 12h timeframe stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Choppiness Index (14) for regime filter
    def calculate_chop(high, low, close, period=14):
        atr_sum = np.zeros(len(close))
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]  # First TR is just high-low
        for i in range(1, len(tr)):
            atr_sum[i] = atr_sum[i-1] + tr[i]
        # Wilder's smoothing: ATR = (prev_ATR * (period-1) + TR) / period
        atr_wilder = np.zeros(len(close))
        atr_wilder[period-1] = np.mean(tr[:period])
        for i in range(period, len(atr_wilder)):
            atr_wilder[i] = (atr_wilder[i-1] * (period-1) + tr[i]) / period
        # Chop = 100 * log10(ATR_sum / (max_high - min_low)) / log10(period)
        max_high = np.zeros(len(close))
        min_low = np.zeros(len(close))
        max_high[period-1] = np.max(high[:period])
        min_low[period-1] = np.min(low[:period])
        for i in range(period, len(max_high)):
            max_high[i] = max(max_high[i-1], high[i])
            min_low[i] = min(min_low[i-1], low[i])
        chop = np.full(len(close), 50.0)  # Default to neutral
        for i in range(period, len(close)):
            if max_high[i] > min_low[i]:
                chop[i] = 100 * np.log10(atr_wilder[i] * period / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_filter = chop > 61.8  # Only trade in ranging markets (chop > 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for EMA, ATR, and Chop
    
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
        curr_ema = ema_34_aligned[i]
        curr_atr = atr[i]
        curr_chop_filter = chop_filter[i]
        
        # Volume confirmation: volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (1.5 * vol_ma_20)
        else:
            volume_confirm = False
        
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
            # Long: price breaks above Camarilla R3, price above 1d EMA34, volume spike, chop filter, in session
            if (curr_close > camarilla_r3 and 
                curr_close > curr_ema and 
                volume_confirm and 
                curr_chop_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla S3, price below 1d EMA34, volume spike, chop filter, in session
            elif (curr_close < camarilla_s3 and 
                  curr_close < curr_ema and 
                  volume_confirm and 
                  curr_chop_filter):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Camarilla S3 OR stoploss hit
            if (curr_close < camarilla_s3 or 
                curr_close < entry_price - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Camarilla R3 OR stoploss hit
            if (curr_close > camarilla_r3 or 
                curr_close > entry_price + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals