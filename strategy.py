# 1D_KAMA_RSI_Chop_Momentum
# Hypothesis: On daily timeframe, KAMA captures trend direction while RSI identifies momentum extremes.
# In trending regimes (Choppiness Index < 38.2), we follow KAMA direction with RSI pullback entries.
# In ranging regimes (Choppiness Index > 61.8), we fade extremes at Bollinger Bands with RSI divergence.
# This dual-regime approach adapts to both bull and bear markets by avoiding false signals in chop
# and capturing trends when they emerge. Uses 1d for signals, 1w for trend filter to reduce whipsaw.
# Target: 30-100 trades over 4 years (7-25/year) with disciplined entries.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA calculation (ER=10, Fast=2, Slow=30) ===
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # needs correction
    # Recalculate volatility properly
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Actually, ER needs to be calculated per period
    # Let's do it correctly
    er = np.zeros_like(close)
    for i in range(1, len(close)):
        if i >= 10:
            change_val = np.abs(close[i] - close[i-10])
            volatility_val = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility_val != 0:
                er[i] = change_val / volatility_val
            else:
                er[i] = 0
        else:
            er[i] = 0
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Bollinger Bands (20, 2) ===
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # === Choppiness Index (14) ===
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    tr1 = true_range(high[1:], low[1:], close[:-1])
    tr = np.concatenate([[0], tr1])  # first TR is high-low
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close)
    for i in range(len(close)):
        if atr_sum[i] > 0 and hh[i] != ll[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # === 1W trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Align indicators
    kama_aligned = kama  # already calculated on close
    rsi_aligned = rsi
    bb_upper_aligned = bb_upper
    bb_lower_aligned = bb_lower
    chop_aligned = chop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        chop_val = chop_aligned[i]
        ema_1w = ema_1w_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short
            if price > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry logic
        if position == 0:
            # Trending regime (chop < 38.2): follow KAMA with RSI pullback
            if chop_val < 38.2:
                # Long: price above KAMA and RSI pulling back from overbought
                if price > kama_val and 40 <= rsi_val <= 50:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short: price below KAMA and RSI pulling back from oversold
                elif price < kama_val and 50 <= rsi_val <= 60:
                    signals[i] = -0.25
                    position = -1
                    continue
            # Ranging regime (chop > 61.8): fade extremes at BB with RSI divergence
            elif chop_val > 61.8:
                # Long: price at lower BB and RSI showing bullish divergence (RSI rising from oversold)
                if price <= bb_lower_val and rsi_val < 40:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short: price at upper BB and RSI showing bearish divergence (RSI falling from overbought)
                elif price >= bb_upper_val and rsi_val > 60:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1D_KAMA_RSI_Chop_Momentum"
timeframe = "1d"
leverage = 1.0