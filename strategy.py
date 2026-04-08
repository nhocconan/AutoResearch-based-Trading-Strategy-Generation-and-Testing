#!/usr/bin/env python3
# 4h_rsi_cci_confluence_v1
# Hypothesis: On 4h timeframe, combine RSI(14) and CCI(20) for mean-reversion entries during low volatility periods.
# Long when RSI < 30 and CCI < -100 with volatility below median.
# Short when RSI > 70 and CCI > 100 with volatility below median.
# Exit when RSI crosses 50 or volatility expands above 1.5x median.
# Uses daily trend filter: only trade in direction of daily EMA50 to avoid counter-trend in strong trends.
# Designed for low frequency (20-40 trades/year) to minimize fee drag and work in both bull/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_cci_confluence_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # CCI(20) calculation
    typical_price = (high + low + close) / 3
    ma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - ma_tp) / (0.015 * mad + 1e-10)
    cci = cci.fillna(0).values
    
    # Volatility filter: ATR(14) normalized by price
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    vol_norm = atr / close
    
    # Daily trend filter: EMA50
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema50_4h = align_htf_to_ltf(prices, df_daily, daily_ema50)
    
    # Median volatility for filter
    vol_median = np.nanmedian(vol_norm[20:]) if np.sum(~np.isnan(vol_norm[20:])) > 0 else 0.01
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi[i]) or np.isnan(cci[i]) or np.isnan(daily_ema50_4h[i]) or np.isnan(vol_norm[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 or volatility expands
            if rsi[i] >= 50 or vol_norm[i] > 1.5 * vol_median:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 or volatility expands
            if rsi[i] <= 50 or vol_norm[i] > 1.5 * vol_median:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volatility filter: only trade in low volatility
            vol_ok = vol_norm[i] < vol_median
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema50_4h[i]
            daily_downtrend = close[i] < daily_ema50_4h[i]
            
            # Long entry: oversold conditions with volatility filter and uptrend bias
            if rsi[i] < 30 and cci[i] < -100 and vol_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: overbought conditions with volatility filter and downtrend bias
            elif rsi[i] > 70 and cci[i] > 100 and vol_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals