# 12h_KAMA_1dRSI_Chop - KAMA direction with 1d RSI and Chop filter
# Trend following with regime filter to avoid whipsaws. KAMA adapts to market conditions,
# RSI avoids overbought/oversold extremes, Chop filter identifies trending vs ranging markets.
# Target: 12-37 trades/year (50-150 total over 4 years)

name = "12h_KAMA_1dRSI_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for RSI and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - adaptive trend
    def calculate_kama(price, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=0)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        
        # Smoothing constants
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(price)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    kama_rising = np.zeros_like(kama, dtype=bool)
    kama_falling = np.zeros_like(kama, dtype=bool)
    kama_rising[1:] = kama[1:] > kama[:-1]
    kama_falling[1:] = kama[1:] < kama[:-1]
    
    # 1d RSI for overbought/oversold
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close_1d)
    rs[14:] = avg_gain[14:] / np.maximum(avg_loss[14:], 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Choppiness Index (14-period) for regime filter
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], 
                     abs(high[i] - close[i-1]), 
                     abs(low[i] - close[i-1]))
            if i == 1:
                atr[i] = tr
            else:
                atr[i] = (atr[i-1] * (period-1) + tr) / period
        
        # Sum of true ranges
        atr_sum = np.zeros(len(close))
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros(len(close))
        ll = np.zeros(len(close))
        for i in range(period-1, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.zeros(len(close))
        for i in range(period-1, len(close)):
            if atr_sum[i] > 0 and (hh[i] - ll[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    chop = calculate_chop(high, low, close, period=14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # Chop filter: trending when Chop < 38.2 or > 61.8
    chop_trending = (chop_aligned < 38.2) | (chop_aligned > 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising, RSI not overbought (<70), trending market
            long_cond = kama_rising[i] and (rsi_aligned[i] < 70) and chop_trending[i]
            # Short: KAMA falling, RSI not oversold (>30), trending market
            short_cond = kama_falling[i] and (rsi_aligned[i] > 30) and chop_trending[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA falling or RSI overbought (>70)
            if kama_falling[i] or (rsi_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA rising or RSI oversold (<30)
            if kama_rising[i] or (rsi_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals