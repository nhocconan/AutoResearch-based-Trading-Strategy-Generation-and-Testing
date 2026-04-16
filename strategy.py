#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) with 4h trend filter (HMA21) and 1d volume confirmation.
# Long when: RSI(14) < 30 (oversold) AND price > 4h HMA21 (uptrend) AND 1d volume > 1.2x 20-period average.
# Short when: RSI(14) > 70 (overbought) AND price < 4h HMA21 (downtrend) AND 1d volume > 1.2x 20-period average.
# Exit on RSI mean reversion (RSI > 50 for long, RSI < 50 for short) or opposite signal.
# Uses discrete position size 0.20. Designed for 1h timeframe with HTF filters to reduce noise and overtrading.
# Target: 60-150 total trades over 4 years (15-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: RSI(14) ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral RSI when no loss
    
    # === 4h Indicators: HMA21 for trend ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # HMA: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_4h).ewm(span=half_len, adjust=False, min_periods=half_len).mean()
    wma_full = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_4h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # === 1d Indicators: Volume Spike (volume > 1.2x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.2 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 21 periods for HMA, 14 for RSI, 20 for volume MA)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        rsi_val = rsi[i]
        hma_val = hma_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI reverts above 50 (mean reversion)
            if rsi_val > 50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI reverts below 50 (mean reversion)
            if rsi_val < 50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: RSI < 30 (oversold) AND price > 4h HMA21 (uptrend) AND volume spike
            if rsi_val < 30 and price > hma_val and vol_spike:
                signals[i] = 0.20
                position = 1
            
            # SHORT: RSI > 70 (overbought) AND price < 4h HMA21 (downtrend) AND volume spike
            elif rsi_val > 70 and price < hma_val and vol_spike:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_RSI14_4hHMA21_1dVolume_V1"
timeframe = "1h"
leverage = 1.0