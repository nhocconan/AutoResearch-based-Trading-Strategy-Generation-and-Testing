#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Uses 1d EMA34 for trend direction and 1d ATR for volatility filtering.
# Long: price breaks above 12h Donchian(20) high AND closes above 1d Camarilla R3 AND 1d uptrend AND volume spike.
# Short: price breaks below 12h Donchian(20) low AND closes below 1d Camarilla S3 AND 1d downtrend AND volume spike.
# Exits on Donchian reversal. Discrete sizing 0.25 balances return and drawdown.
# Target: 50-150 total trades over 4 years (12-37/year) on BTC/ETH/SOL.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR(14) for volatility filter (avoid low volatility chop)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3) - strong breakout levels
    # Camarilla: based on previous day's high, low, close
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    prev_daily_high = df_1d['high'].shift(1).values
    prev_daily_low = df_1d['low'].shift(1).values
    prev_daily_close = df_1d['close'].shift(1).values
    
    camarilla_r3 = prev_daily_close + (prev_daily_high - prev_daily_low) * 1.1 / 4
    camarilla_s3 = prev_daily_close - (prev_daily_high - prev_daily_low) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 12h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume on 12h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    # ATR filter: avoid trading when volatility is too low (choppy market)
    # Only trade when 1d ATR > 0.5 * 50-period average ATR (ensures sufficient volatility)
    atr_ma_50_1d = pd.Series(atr_14_1d_aligned).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_14_1d_aligned > (atr_ma_50_1d * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20, 50) + 1  # 51 (for EMA34, Donchian20, ATR MA50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_ma_20[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(atr_ma_50_1d[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Volatility filter: avoid low volatility chop
        vol_filter_ok = vol_filter[i]
        
        # Donchian breakout conditions
        breakout_up = curr_high > donchian_high[i]  # Break above upper Donchian
        breakdown_down = curr_low < donchian_low[i]  # Break below lower Donchian
        
        # Daily Camarilla R3/S3 confirmation
        breakout_r3 = curr_close > camarilla_r3_aligned[i]  # Confirm above daily R3
        breakdown_s3 = curr_close < camarilla_s3_aligned[i]  # Confirm below daily S3
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND daily R3 confirmation AND uptrend AND volume confirmation AND vol filter
            if breakout_up and breakout_r3 and uptrend and vol_confirm and vol_filter_ok:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown down AND daily S3 confirmation AND downtrend AND volume confirmation AND vol filter
            elif breakdown_down and breakdown_s3 and downtrend and vol_confirm and vol_filter_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown (reversal signal)
            if curr_low < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout (reversal signal)
            if curr_high > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals