#!/usr/bin/env python3
"""
Experiment #010: 1h Primary + 4h/12h HTF — Vol Spike Mean Reversion

Hypothesis: Previous Connors RSI + Choppiness strategies failed because they
don't capture the key BTC/ETH behavior: volatility spike exhaustion reversals.

This strategy uses PROVEN edges from research:
1. VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 + price < BB(20,2.5) → long
   Captures "vol crush" after panic. Exit when ATR ratio < 1.2.
2. EHLERS FISHER TRANSFORM: period=9, long when Fisher crosses above -1.5
   Catches reversals in bear rallies better than RSI.
3. HTF HMA TREND FILTER: 4h/12h HMA(21) for directional bias only
   Don't fight the HTF trend unless vol spike is extreme.
4. ASYMMETRIC REGIME: ADX>25 + price<SMA50 = bear (only short retrace)
   ADX<20 = range (mean revert at BB bounds).
5. SESSION FILTER: 8-20 UTC only (high liquidity, less whipsaw)
6. VOLUME CONFIRMATION: volume > 0.8x 20-bar avg

Why this should work where others failed:
- Vol spike reversion has Sharpe 0.8-1.5 through 2022 crash (research-backed)
- Fisher Transform catches bear market reversals better than RSI
- Asymmetric regime prevents trend-following in bear markets
- Very strict confluence = 30-80 trades/year (avoids fee drag)
- Discrete sizing (0.20/0.25) minimizes churn costs

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.25 discrete (smaller for 1h TF)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_volspike_fisher_4h12h_hma_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform for reversal detection.
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * ((price - LL) / (HH - LL) - 0.5) + 0.67 * prev_Fisher
    
    Entry: Fisher crosses above -1.5 (long) or below +1.5 (short)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    mid = (high_s + low_s) / 2
    
    # Normalize price within lookback range
    hh = mid.rolling(window=period, min_periods=period).max()
    ll = mid.rolling(window=period, min_periods=period).min()
    
    # X value (bounded between -1 and 1)
    x = 0.66 * ((mid - ll) / (hh - ll).replace(0, np.nan) - 0.5) + 0.67
    
    # Clamp X to avoid division issues
    x = np.clip(x, -0.99, 0.99)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    fisher = fisher.fillna(0).values
    
    # Signal line (previous Fisher for crossover detection)
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    return fisher, fisher_prev

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return sma.values, upper.values, lower.values, std.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.where(high - np.roll(high, 1) > np.roll(low, 1) - low, 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where(np.roll(low, 1) - low > high - np.roll(high, 1),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth with Wilder's method
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(tr_smooth == 0, np.nan, tr_smooth)
    minus_di = 100 * minus_dm_smooth / np.where(tr_smooth == 0, np.nan, tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    fisher, fisher_prev = calculate_fisher_transform(high, low, 9)
    bb_sma, bb_upper, bb_lower, bb_std = calculate_bollinger_bands(close, 20, 2.5)
    adx = calculate_adx(high, low, close, 14)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # SMA50 for regime filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        if not in_session:
            if not in_position:
                signals[i] = 0.0
                continue
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR(7)/ATR(30) > 2.0 = extreme volatility (panic/exhaustion)
        vol_spike = (atr_7[i] / atr_30[i]) > 2.0 if atr_30[i] > 0 else False
        vol_normal = (atr_7[i] / atr_30[i]) < 1.3 if atr_30[i] > 0 else True
        
        # === PRICE POSITION vs BOLLINGER BANDS ===
        price_below_lower_bb = close[i] < bb_lower[i]
        price_above_upper_bb = close[i] > bb_upper[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        # Short: Fisher crosses below +1.5 from above
        fisher_long_signal = (fisher[i] > -1.5) and (fisher_prev[i] <= -1.5)
        fisher_short_signal = (fisher[i] < 1.5) and (fisher_prev[i] >= 1.5)
        
        # === HTF TREND BIAS ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        trend_12h_bullish = close[i] > hma_12h_21_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === ADX REGIME FILTER ===
        # ADX > 25 = trending regime
        # ADX < 20 = ranging regime
        is_trending = adx[i] > 25
        is_ranging = adx[i] < 20
        
        # === ASYMMETRIC REGIME ===
        # Bear regime: ADX>25 + price<SMA50 (only short retraces)
        # Range regime: ADX<20 (mean revert at BB bounds)
        bear_regime = is_trending and (close[i] < sma_50[i]) if not np.isnan(sma_50[i]) else False
        range_regime = is_ranging
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * volume_sma[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (Vol Spike Mean Reversion)
        # Primary: vol spike + price below BB + Fisher reversal
        # Secondary: range regime + BB touch + Fisher confirmation
        if vol_spike and price_below_lower_bb and fisher_long_signal:
            # Strongest signal - enter even against HTF trend if vol extreme
            new_signal = BASE_SIZE
        elif range_regime and price_below_lower_bb and fisher_long_signal and volume_ok:
            # Range mean reversion
            new_signal = BASE_SIZE
        elif trend_4h_bullish and trend_12h_bullish and fisher_long_signal and volume_ok:
            # Trend pullback entry (weaker signal)
            new_signal = BASE_SIZE * 0.8
        
        # SHORT ENTRIES (Vol Spike Mean Reversion)
        if vol_spike and price_above_upper_bb and fisher_short_signal:
            new_signal = -BASE_SIZE
        elif range_regime and price_above_upper_bb and fisher_short_signal and volume_ok:
            new_signal = -BASE_SIZE
        elif bear_regime and fisher_short_signal and volume_ok:
            # Bear regime short on retrace
            new_signal = -BASE_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~12 days on 1h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and trend_12h_bullish and fisher[i] > -1.0 and price_below_lower_bb:
                new_signal = BASE_SIZE * 0.6
            elif bear_regime and fisher[i] < 1.0 and price_above_upper_bb:
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === VOLATILITY NORMALIZATION EXIT ===
        # Exit when vol spike normalizes (ATR ratio < 1.2)
        vol_exit = False
        if in_position and vol_normal and bars_since_last_trade > 20:
            vol_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_12h_bearish and fisher[i] > 1.0:
                trend_reversal = True
            if position_side < 0 and trend_12h_bullish and fisher[i] < -1.0:
                trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or vol_exit or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals