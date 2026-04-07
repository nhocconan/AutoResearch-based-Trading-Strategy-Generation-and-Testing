# 1h Volume Weighted RSI with 4h/1d Trend Filter
# Hypothesis: RSI(14) pulled back to 30-40 (oversold) or 60-70 (overbought) on 1h
# with volume confirmation, aligned with 4h trend (EMA50) and 1d regime (ADX<25 for mean reversion).
# Works in bull/bear: buys dips in uptrend, sells rallies in downtrend, avoids chop.
# Target: 15-30 trades/year (60-120 over 4 years).

name = "1h_vw_rsi_4h1d_trend_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for trend and regime filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h = np.roll(ema_50_4h, 1)
    if len(ema_50_4h) > 1:
        ema_50_4h[0] = ema_50_4h[1]
    else:
        ema_50_4h[0] = 0
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d ADX(14) for regime filter (avoid chop)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.zeros_like(arr)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_1d = smooth_wilder(tr, 14)
    plus_di_1d = 100 * smooth_wilder(plus_dm, 14) / (atr_1d + 1e-10)
    minus_di_1d = 100 * smooth_wilder(minus_dm, 14) / (atr_1d + 1e-10)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = smooth_wilder(dx_1d, 14)
    adx_1d = np.roll(adx_1d, 1)
    if len(adx_1d) > 1:
        adx_1d[0] = adx_1d[1]
    else:
        adx_1d[0] = 25
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1h RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def rsi_wilder(gain, loss, period=14):
        rsi = np.zeros_like(close)
        if len(gain) < period:
            return rsi
        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])
        if avg_loss == 0:
            rsi[period] = 100
        else:
            rsi[period] = 100 - (100 / (1 + avg_gain / avg_loss))
        for i in range(period+1, len(close)):
            avg_gain = (avg_gain * (period-1) + gain[i-1]) / period
            avg_loss = (avg_loss * (period-1) + loss[i-1]) / period
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rsi[i] = 100 - (100 / (1 + avg_gain / avg_loss))
        return rsi
    
    rsi_1h = rsi_wilder(gain, loss, 14)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(rsi_1h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when not too choppy (ADX < 25 = range, good for mean reversion)
        if adx_1d_aligned[i] > 25:
            # In trending markets, reduce activity or go flat
            if position != 0:
                # Exit if trend changes against position
                if position == 1 and close[i] < ema_50_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and close[i] > ema_50_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20 if position == 1 else -0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 50 or trend fails
            if rsi_1h[i] > 50 or close[i] < ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit: RSI < 50 or trend fails
            if rsi_1h[i] < 50 or close[i] > ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Long entry: RSI oversold (30-40) with volume and above 4h EMA50
            if (30 <= rsi_1h[i] <= 40 and vol_filter[i] and close[i] > ema_50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: RSI overbought (60-70) with volume and below 4h EMA50
            elif (60 <= rsi_1h[i] <= 70 and vol_filter[i] and close[i] < ema_50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals