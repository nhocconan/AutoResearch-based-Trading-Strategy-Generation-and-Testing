# 1d_iceberg_breakout_v1
# Detects institutional iceberg orders via volume anomalies at key levels
# Uses: 1d price action + 1w trend filter + volume spike detection
# Works in bull/bear: accumulation (bull) and distribution (bear) both show volume anomalies
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_iceberg_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend direction
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Daily ATR(14) for volatility and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Volume anomaly detection: current volume > 2.5x 20-day average
    vol_ma20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma20[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (vol_ma20 * 2.5)
    
    # Price channels: Donchian(20) for breakout levels
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_21_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * atr[i]
            
            # Exit: price breaks below Donchian low (failed breakout) or stoploss
            if (close[i] < donch_low[i] or close[i] < stop_loss):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * atr[i]
            
            # Exit: price breaks above Donchian high (failed breakdown) or stoploss
            if (close[i] > donch_high[i] or close[i] > stop_loss):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            if volume_spike[i]:
                # Volume spike + breakout above Donchian high in uptrend
                if (close[i] > donch_high[i] and close[i-1] <= donch_high[i] and 
                    close[i] > ema_21_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Volume spike + breakdown below Donchian low in downtrend
                elif (close[i] < donch_low[i] and close[i-1] >= donch_low[i] and 
                      close[i] < ema_21_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals