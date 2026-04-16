#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h EMA(50) trend filter and volume confirmation.
# Long when RSI < 30 AND price > 4h EMA(50) (uptrend) AND volume > 1.2x 20-period average.
# Short when RSI > 70 AND price < 4h EMA(50) (downtrend) AND volume > 1.2x 20-period average.
# Exit on opposite RSI extreme (RSI > 50 for longs, RSI < 50 for shorts) or ATR-based stop (1.5*ATR).
# Uses discrete position size 0.20. Designed to capture mean reversion in trending markets with volume confirmation.
# Uses 4h for trend direction and volume confirmation, 1h only for entry timing via RSI extremes.
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: EMA(50) for trend and volume MA ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # EMA(50) for trend
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume MA(20) for confirmation
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    volume_spike = volume > (1.2 * vol_ma_4h_aligned)
    
    # === 1h Indicators: RSI(14) and ATR(14) for stoploss ===
    # RSI calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR calculation
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA/RSI/ATR)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(rsi[i]) or
            np.isnan(atr[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        price_vs_ema = price > ema_4h_aligned[i]  # True if price above 4h EMA (uptrend bias)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI > 50 (mean reversion complete) OR ATR stop hit
            if rsi_val > 50 or price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI < 50 (mean reversion complete) OR ATR stop hit
            if rsi_val < 50 or price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: RSI < 30 (oversold) AND price > 4h EMA (uptrend) AND volume spike
            if rsi_val < 30 and price_vs_ema and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: RSI > 70 (overbought) AND price < 4h EMA (downtrend) AND volume spike
            elif rsi_val > 70 and not price_vs_ema and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_RSI14_4hEMA50_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0