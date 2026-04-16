#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with weekly trend filter and volume confirmation.
# Long when price breaks above R4 AND weekly close > weekly open (bullish weekly candle) AND volume > 1.5x 20-period average.
# Short when price breaks below S4 AND weekly close < weekly open (bearish weekly candle) AND volume > 1.5x 20-period average.
# Exit on opposite Camarilla level (R3 for longs, S3 for shorts) or ATR-based stop (2*ATR from entry).
# Uses discrete position size 0.25. Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation avoids false breakouts. Designed for 6h timeframe to capture multi-day moves with controlled trade frequency.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Camarilla Pivot Levels (based on prior 6h bar) ===
    # Camarilla levels calculated from prior bar's high, low, close
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1 / 4)
    # S3 = close - ((high - low) * 1.1 / 4)
    # S4 = close - ((high - low) * 1.1 / 2)
    # Using prior bar values to avoid look-ahead
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    prior_high[0] = prior_low[0] = prior_close[0] = np.nan  # first bar has no prior
    
    camarilla_r4 = prior_close + ((prior_high - prior_low) * 1.1 / 2)
    camarilla_r3 = prior_close + ((prior_high - prior_low) * 1.1 / 4)
    camarilla_s3 = prior_close - ((prior_high - prior_low) * 1.1 / 4)
    camarilla_s4 = prior_close - ((prior_high - prior_low) * 1.1 / 2)
    
    # === 1w Indicators: Weekly trend filter (bullish/bearish weekly candle) ===
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # bullish weekly candle
    weekly_bearish = weekly_close < weekly_open  # bearish weekly candle
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # === 1w Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    weekly_volume = df_1w['volume'].values
    vol_ma_1w = pd.Series(weekly_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    volume_spike = volume > (1.5 * vol_ma_1w_aligned)
    
    # === 6h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ATR/volume MA)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_s4[i]) or np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_6h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S3 (profit target/reversal)
            if price < camarilla_s3[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R3 (profit target/reversal)
            if price > camarilla_r3[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R4 AND weekly bullish candle AND volume spike
            if price > camarilla_r4[i] and weekly_bullish_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S4 AND weekly bearish candle AND volume spike
            elif price < camarilla_s4[i] and weekly_bearish_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R4_S4_1wTrend_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0