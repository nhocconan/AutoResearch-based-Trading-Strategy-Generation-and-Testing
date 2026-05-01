#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams Alligator with 1d trend filter and volume confirmation.
# Elder Ray measures bull/bear power (close-EMA13) to gauge trend strength.
# Williams Alligator (jaw/teeth/lips) identifies trend and entry zones.
# Combined, they filter for strong momentum with pullback entries in the direction of 1d trend.
# Designed for low trade frequency (<150/year) to avoid fee drag, works in bull/bear via 1d trend alignment.

name = "6h_ElderRay_Alligator_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Calculate 1d ADX(14) for trend strength filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.diff(high, prepend=high[0])
        minus_dm = np.diff(low, prepend=low[0]) * -1
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        tr = np.maximum(np.abs(np.diff(high, prepend=high[0])),
                        np.maximum(np.abs(np.diff(low, prepend=low[0])),
                                   np.abs(np.diff(close, prepend=close[0]))))
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate Williams Alligator on 6h
    # Jaw: 13-period SMMA smoothed 8 periods ahead
    # Teeth: 8-period SMMA smoothed 5 periods ahead  
    # Lips: 5-period SMMA smoothed 3 periods ahead
    def smma(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) >= period:
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 13, 20) + 5  # Ensure all indicators are valid
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(ema_13[i]) or
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # 1d trend and strength filter
        uptrend_1d = curr_close > ema_34_1d_aligned[i]
        downtrend_1d = curr_close < ema_34_1d_aligned[i]
        strong_trend = adx_1d_aligned[i] > 25
        
        # Elder Ray: bull/bear power confirmation
        bull_power_confirm = bull_power[i] > 0  # Bullish momentum
        bear_power_confirm = bear_power[i] < 0  # Bearish momentum
        
        # Williams Alligator: alignment and entry signals
        # Alligator aligned: Jaw > Teeth > Lips (downtrend) or Lips > Teeth > Jaw (uptrend)
        alligator_aligned_down = jaw[i] > teeth[i] and teeth[i] > lips[i]
        alligator_aligned_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        
        # Price relative to Alligator for entry
        price_above_lips = curr_close > lips[i]
        price_below_lips = curr_close < lips[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: 1d uptrend + strong trend + bull power + Alligator aligned up + price above lips + volume
            if (uptrend_1d and strong_trend and bull_power_confirm and 
                alligator_aligned_up and price_above_lips and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: 1d downtrend + strong trend + bear power + Alligator aligned down + price below lips + volume
            elif (downtrend_1d and strong_trend and bear_power_confirm and 
                  alligator_aligned_down and price_below_lips and vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bear power reversal or price below teeth (trend weakening)
            if bear_power[i] >= 0 or curr_close < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bull power reversal or price above teeth (trend weakening)
            if bull_power[i] <= 0 or curr_close > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals