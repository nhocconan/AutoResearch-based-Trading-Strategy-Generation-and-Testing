#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w KAMA trend with RSI mean reversion and chop regime filter
# Weekly KAMA determines primary trend direction (bull/bear/range)
# Daily RSI(14) provides mean reversion entries in direction of weekly trend
# Daily choppiness index (CHOP) filters ranging vs trending markets: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (follow momentum)
# In ranging markets (CHOP > 61.8): fade extreme RSI (<30 long, >70 short)
# In trending markets (CHOP < 38.2): pullback to RSI 50 in direction of weekly KAMA trend
# Volume confirmation: current 1d volume > 1.2x 20-day average
# Position size: 0.25 discrete levels to minimize fee churn
# Target: 20-60 trades/year on 1d timeframe (80-240 total over 4 years)

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio (ER) = |Change| / Sum|Daily Changes|
    change_1w = np.abs(np.diff(close_1w))
    abs_change_1w = np.abs(np.diff(close_1w))
    er_1w = np.zeros_like(close_1w)
    for i in range(10, len(close_1w)):
        if i >= 10:
            net_change = np.abs(close_1w[i] - close_1w[i-9])
            total_change = np.sum(np.abs(np.diff(close_1w[i-9:i+1])))
            if total_change > 0:
                er_1w[i] = net_change / total_change
            else:
                er_1w[i] = 0
    er_1w[0:10] = 0
    
    # Smoothing constants: fastest EMA = 2/(2+1) = 0.67, slowest = 2/(30+1) = 0.0645
    sc_1w = (er_1w * (0.67 - 0.0645) + 0.0645) ** 2
    
    # Calculate KAMA
    kama_1w = np.zeros_like(close_1w)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    
    # Calculate 1w ATR (14-period) for volatility filtering
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1w indicators to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Pre-compute daily indicators
    # Daily RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi[0:14] = 50
    
    # Daily Choppiness Index (CHOP) - measures if market is ranging or trending
    # CHOP = 100 * LOG10(SUM(TR(14)) / (LOG10(N) * (MAX(HIGH)-LOW(14)))) / LOG10(N)
    # Simplified: CHOP = 100 * log10(sum(atr14) / (log10(14) * (max(high14)-min(low14)))) / log10(14)
    atr_14_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values  # TR already calculated above
    sum_atr_14 = pd.Series(atr_14_daily).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop = np.zeros_like(close)
    for i in range(13, len(close)):
        if sum_atr_14[i] > 0 and range_14[i] > 0:
            chop[i] = 100 * np.log10(sum_atr_14[i] / (np.log10(14) * range_14[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no data
    chop[0:13] = 50
    
    # Volume confirmation (20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(atr_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(vol_ma_20[i]) or
            atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.2x 20-day average
        volume_confirmed = volume[i] > 1.2 * vol_ma_20[i]
        
        # Volatility filter: only trade when ATR is above its 50-day average (avoid low-vol chop)
        atr_ma_50 = pd.Series(atr_aligned).rolling(window=50, min_periods=50).mean()
        if len(atr_ma_50) > i:
            vol_filter = atr_aligned[i] > atr_ma_50.iloc[i]
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size: 0.25 (25% of capital) - discrete level to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: RSI > 70 (overbought) or chop regime shift to strong trend
            if rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            elif chop[i] < 30 and close[i] < kama_aligned[i]:  # Strong trend but price below KAMA
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: RSI < 30 (oversold) or chop regime shift to strong trend
            if rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            elif chop[i] < 30 and close[i] > kama_aligned[i]:  # Strong trend but price above KAMA
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic based on chop regime and RSI
            if volume_confirmed:
                if chop[i] > 61.8:  # Ranging market - mean reversion
                    # Fade extreme RSI: long when oversold, short when overbought
                    if rsi[i] < 30 and close[i] > kama_aligned[i]:  # Long: oversold but above weekly KAMA (bullish bias)
                        position = 1
                        signals[i] = position_size
                    elif rsi[i] > 70 and close[i] < kama_aligned[i]:  # Short: overbought but below weekly KAMA (bearish bias)
                        position = -1
                        signals[i] = -position_size
                elif chop[i] < 38.2:  # Trending market - follow momentum
                    # Pullback to RSI 50 in direction of weekly KAMA trend
                    if close[i] > kama_aligned[i] and rsi[i] < 50 and rsi[i] > 30:  # Long: above KAMA, RSI pulling back from oversold
                        position = 1
                        signals[i] = position_size
                    elif close[i] < kama_aligned[i] and rsi[i] > 50 and rsi[i] < 70:  # Short: below KAMA, RSI pulling back from overbought
                        position = -1
                        signals[i] = -position_size
    
    return signals