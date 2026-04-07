#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Momentum with 4h Trend Filter and Volume Confirmation
# Hypothesis: 1h momentum bursts aligned with 4h trend capture directional moves
# while avoiding counter-trend trades. Volume confirmation ensures institutional
# participation. Session filter (08-20 UTC) reduces noise. Targets 15-35 trades/year.
# Works in bull/bear markets by only taking trades aligned with higher timeframe trend.

name = "1h_momentum_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.nanmean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (np.zeros_like(high))
        minus_di = 100 * (np.zeros_like(high))
        
        plus_sm = np.zeros_like(high)
        minus_sm = np.zeros_like(high)
        plus_sm[period] = np.nansum(plus_dm[1:period+1])
        minus_sm[period] = np.nansum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            plus_sm[i] = (plus_sm[i-1] * (period-1) + plus_dm[i]) / period
            minus_sm[i] = (minus_sm[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_sm / atr
        minus_di = 100 * minus_sm / atr
        
        dx = np.zeros_like(high)
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        adx = np.zeros_like(high)
        adx[2*period] = np.nanmean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_4h = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1h RSI(14) for momentum
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1h = calculate_rsi(close)
    
    # 1h volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(adx_4h_aligned[i]) or 
            np.isnan(rsi_1h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: RSI < 40 OR trend weakens (ADX < 20) OR price below 4h EMA50
            if (rsi_1h[i] < 40 or 
                adx_4h_aligned[i] < 20 or 
                close[i] < ema50_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI > 60 OR trend weakens (ADX < 20) OR price above 4h EMA50
            if (rsi_1h[i] > 60 or 
                adx_4h_aligned[i] < 20 or 
                close[i] > ema50_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Strong trend required: ADX > 25
            if adx_4h_aligned[i] > 25:
                # Long: RSI > 55 + volume confirmation + above 4h EMA50 (uptrend)
                if (rsi_1h[i] > 55 and 
                    vol_confirm and 
                    close[i] > ema50_4h_aligned[i]):
                    position = 1
                    signals[i] = 0.20
                # Short: RSI < 45 + volume confirmation + below 4h EMA50 (downtrend)
                elif (rsi_1h[i] < 45 and 
                      vol_confirm and 
                      close[i] < ema50_4h_aligned[i]):
                    position = -1
                    signals[i] = -0.20
    
    return signals