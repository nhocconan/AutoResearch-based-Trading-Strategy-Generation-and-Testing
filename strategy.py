# 12h_VWAP_Trend_VolumeRegime
# Hypothesis: 12h VWAP trend following with 1d volume confirmation and 1d volatility regime filter.
# Long when price > VWAP, volume above average, and low volatility regime (ATR ratio < 1.0).
# Short when price < VWAP, volume above average, and low volatility regime.
# Uses VWAP for trend, volume for conviction, and volatility filter to avoid choppy markets.
# Designed for 12h timeframe to target 50-150 trades over 4 years with low frequency.

name = "12h_VWAP_Trend_VolumeRegime"
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
    
    # Get daily data once for volume and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h VWAP
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # Daily average volume (20-period)
    daily_vol = df_1d['volume'].values
    avg_vol_20 = np.convolve(daily_vol, np.ones(20)/20, mode='same')
    # Handle edges
    avg_vol_20[:10] = np.mean(daily_vol[:20]) if len(daily_vol) >= 20 else np.mean(daily_vol)
    avg_vol_20[-10:] = np.mean(daily_vol[-20:]) if len(daily_vol) >= 20 else np.mean(daily_vol)
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    # Daily ATR (14-period) for volatility regime
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    tr1 = np.abs(daily_high - daily_low)
    tr2 = np.abs(np.diff(daily_close, prepend=daily_close[0]))
    tr3 = np.abs(np.diff(daily_close, prepend=daily_close[0]))
    tr3 = np.abs(np.roll(daily_close, 1) - daily_close)
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Daily ATR ratio (current / 20-period average) for regime filter
    atr_ma_20 = np.convolve(atr_14, np.ones(20)/20, mode='same')
    atr_ma_20[:10] = np.mean(atr_14[:20]) if len(atr_14) >= 20 else np.mean(atr_14)
    atr_ma_20[-10:] = np.mean(atr_14[-20:]) if len(atr_14) >= 20 else np.mean(atr_14)
    atr_ratio = atr_14 / atr_ma_20
    atr_ratio = np.where(atr_ma_20 == 0, 1.0, atr_ratio)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap[i]) or np.isnan(avg_vol_20_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        vol_now = volume[i]
        avg_vol = avg_vol_20_aligned[i]
        vol_regime = atr_ratio_aligned[i]
        
        if position == 0:
            # Enter long: price > VWAP, volume > average, low volatility (regime)
            if price > vwap_val and vol_now > avg_vol and vol_regime < 1.0:
                signals[i] = 0.25
                position = 1
            # Enter short: price < VWAP, volume > average, low volatility (regime)
            elif price < vwap_val and vol_now > avg_vol and vol_regime < 1.0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < VWAP or high volatility regime
            if price < vwap_val or vol_regime > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > VWAP or high volatility regime
            if price > vwap_val or vol_regime > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals