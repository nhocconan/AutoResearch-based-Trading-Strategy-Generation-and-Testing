#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
# Uses 4h EMA34 for stable trend direction and 1d volatility regime filter.
# Requires volume > 1.8x 20-period average and trades only during 08-20 UTC session.
# Designed for low trade frequency (~20-40 trades/year) to minimize fee drag.
# Camarilla levels provide intraday support/resistance that work in both trending and ranging markets.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_VolumeConfirm_v1"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Load 1d data ONCE before loop for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(20) for volatility regime filter
    if len(df_1d) >= 20:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1d = pd.Series(tr_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
        
        # Calculate ATR percentile rank (20-period) for regime filter
        atr_rank = np.full(len(df_1d), np.nan)
        for i in range(20, len(atr_1d)):
            if not np.isnan(atr_1d[i]):
                window = atr_1d[i-20:i+1]
                valid_window = window[~np.isnan(window)]
                if len(valid_window) > 0:
                    atr_rank[i] = (np.sum(valid_window <= atr_1d[i]) / len(valid_window)) * 100
        
        atr_1d_rank_aligned = align_htf_to_ltf(prices, df_1d, atr_rank)
    else:
        atr_1d_rank_aligned = np.full(n, np.nan)
    
    # Calculate ATR for stoploss (using 14-period ATR on 1h)
    if n >= 14:
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        atr = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA34 and ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(atr_1d_rank_aligned[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_4h = ema_34_4h_aligned[i]
        curr_atr = atr[i]
        curr_atr_rank = atr_1d_rank_aligned[i]
        
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
        
        # Volume confirmation: volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (1.8 * vol_ma_20)
        else:
            volume_confirm = False
        
        # Volatility regime filter: only trade when ATR rank is between 30 and 70 (avoid extreme volatility)
        vol_regime_ok = (curr_atr_rank >= 30) and (curr_atr_rank <= 70)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, 4h EMA34 uptrend, volume spike, OK volatility regime
            if (curr_close > camarilla_r3 and 
                curr_close > curr_ema_34_4h and 
                volume_confirm and 
                vol_regime_ok):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla S3, 4h EMA34 downtrend, volume spike, OK volatility regime
            elif (curr_close < camarilla_s3 and 
                  curr_close < curr_ema_34_4h and 
                  volume_confirm and 
                  vol_regime_ok):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Camarilla S3, or ATR stoploss hit
            if (curr_close < camarilla_s3) or \
               curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Camarilla R3, or ATR stoploss hit
            if (curr_close > camarilla_r3) or \
               curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals