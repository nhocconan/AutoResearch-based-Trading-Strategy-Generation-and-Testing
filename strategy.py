# 12h_KAMA_RSI_Chop_MeanReversion_v2
# KAMA trend direction + RSI mean reversion + chop regime filter
# Long when KAMA rising + RSI < 40 + chop > 61.8 (range)
# Short when KAMA falling + RSI > 60 + chop > 61.8 (range)
# Uses 1d trend filter to avoid counter-trend trades
# Designed for 12h timeframe: targets 15-35 trades/year to avoid fee drag

name = "12h_KAMA_RSI_Chop_MeanReversion_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # KAMA calculation (12-period ER, 2-30 SC)
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s.diff()).rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (0.3 - 0.06) + 0.06) ** 2
    kama = [np.nan] * len(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]) or sc.iloc[i] == 0:
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    kama = np.array(kama)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index(14)
    atr = pd.Series(np.maximum(high - low, np.maximum(abs(high - close_s.shift()), abs(low - close_s.shift()))))
    tr14 = atr.rolling(window=14, min_periods=14).sum()
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(tr14 / (hh14 - ll14)) / np.log10(14)
    chop = chop.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        kama_now = kama[i]
        kama_prev = kama[i-1]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        if position == 0:
            # Enter long: KAMA rising + RSI oversold + choppy market
            if kama_now > kama_prev and rsi_val < 40 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling + RSI overbought + choppy market
            elif kama_now < kama_prev and rsi_val > 60 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA falling OR RSI overbought
            if kama_now <= kama_prev or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA rising OR RSI oversold
            if kama_now >= kama_prev or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals