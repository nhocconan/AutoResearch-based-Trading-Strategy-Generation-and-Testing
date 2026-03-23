#!/usr/bin/env python3
"""
Experiment #040: 1h Primary + 4h/12h HTF — Fisher Transform + ADX Regime + Session Filter

Hypothesis: 1h timeframe with 4h/12h trend bias, Fisher Transform entries, and ADX regime
will generate 40-80 trades/year with Sharpe > 0.486. Key changes from failed #030, #035:

1) Fisher Transform instead of Connors RSI - cleaner reversal signals, less noise
2) ADX regime instead of Choppiness - more reliable trend/range detection
3) 4h HMA (not 1d) for trend - more responsive, fewer missed entries
4) LOOSE volume filter (0.5x avg, not 0.8x) - ensures trades generate
5) Session filter 8-20 UTC - avoids low-liquidity whipsaws
6) Position size 0.25 (smaller for 1h TF to reduce fee impact)

Why this should work on 1h:
- Fisher Transform catches reversals at extremes (proven on crypto)
- ADX > 25 = trend follow, ADX < 20 = mean revert (clear regime split)
- 4h HMA gives direction without over-filtering (1d was too slow)
- Session filter reduces noise but doesn't block all trades
- LOOSE thresholds ensure 40+ trades/year (avoid Sharpe=0.000 failure)

Position size: 0.25 (discrete, smaller for 1h to minimize fee drag)
Stoploss: 2.0*ATR trailing (tighter for lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_adx_session_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))
    
    for i in range(1, len(close)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=n, min_periods=n, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=n, min_periods=n, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=n, min_periods=n, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=n, min_periods=n, adjust=False).mean().values
    return adx, plus_di, minus_di

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i-period+1:i+1].max() + low[i-period+1:i+1].min()) / 2.0
        
        # Normalize to -1 to +1 range
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        range_val = highest - lowest + 1e-10
        normalized = 2.0 * (close[i] - lowest) / range_val - 1.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        
        # Signal line (1-bar lag)
        if i > 0:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    import datetime
    dt = datetime.datetime.utcfromtimestamp(open_time / 1000.0)
    return dt.hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 12h HMA for macro bias
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    hma_21 = calculate_hma(close, period=21)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Discrete, smaller for 1h TF
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(adx_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(fisher[i]) or np.isnan(bb_upper[i]) or np.isnan(hma_21[i]):
            continue
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        is_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER (LOOSE - 0.5x avg) ===
        volume_ratio = volume[i] / (vol_ma_20[i] + 1e-10)
        has_volume = volume_ratio > 0.5  # Very loose to ensure trades
        
        # === 4H/12H TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === ADX REGIME ===
        adx_value = adx_14[i]
        is_trending = adx_value > 25.0  # Trend market
        is_ranging = adx_value < 20.0  # Range market (with hysteresis)
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5  # Reversal long signal
        fisher_overbought = fisher[i] > 1.5  # Reversal short signal
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher[i-1] <= fisher_signal[i-1] if i > 0 else False
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher[i-1] >= fisher_signal[i-1] if i > 0 else False
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_mid[i] < 0.05  # Tight bands
        
        # === HMA TREND ===
        hma_bullish = close[i] > hma_21[i]
        hma_bearish = close[i] < hma_21[i]
        hma_slope_up = hma_21[i] > hma_21[i-5] if i > 5 else False
        hma_slope_down = hma_21[i] < hma_21[i-5] if i > 5 else False
        
        # === DI DIRECTION ===
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TRENDING REGIME: Follow 4h trend with Fisher entry ---
        if is_trending:
            # Long: 4h bullish + Fisher oversold/cross + session + volume
            if price_above_hma_4h and di_bullish:
                if (fisher_oversold or fisher_cross_up) and is_session and has_volume:
                    if price_above_hma_12h or hma_slope_up:  # 12h confirms or 1h momentum
                        new_signal = POSITION_SIZE
            
            # Short: 4h bearish + Fisher overbought/cross + session + volume
            elif price_below_hma_4h and di_bearish:
                if (fisher_overbought or fisher_cross_down) and is_session and has_volume:
                    if price_below_hma_12h or hma_slope_down:  # 12h confirms or 1h momentum
                        new_signal = -POSITION_SIZE
        
        # --- RANGING REGIME: Mean reversion at BB bounds ---
        elif is_ranging:
            # Long: BB lower + Fisher oversold + 4h not strongly bearish
            if price_below_bb_lower and fisher_oversold:
                if not price_below_hma_12h or fisher_cross_up:  # 12h neutral or Fisher turning
                    if is_session and has_volume:
                        new_signal = POSITION_SIZE
            
            # Short: BB upper + Fisher overbought + 4h not strongly bullish
            elif price_above_bb_upper and fisher_overbought:
                if not price_above_hma_12h or fisher_cross_down:  # 12h neutral or Fisher turning
                    if is_session and has_volume:
                        new_signal = -POSITION_SIZE
        
        # --- FALLBACK: HMA crossover (ensures trades generate) ---
        if new_signal == 0.0:
            # Long: Price crosses above HMA + 4h bullish + session
            if close[i] > hma_21[i] and close[i-1] <= hma_21[i-1]:
                if price_above_hma_4h and is_session and has_volume:
                    new_signal = POSITION_SIZE
            
            # Short: Price crosses below HMA + 4h bearish + session
            elif close[i] < hma_21[i] and close[i-1] >= hma_21[i-1]:
                if price_below_hma_4h and is_session and has_volume:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_4h and adx_value > 25:  # Trend reversed
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h and adx_value > 25:  # Trend reversed
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals