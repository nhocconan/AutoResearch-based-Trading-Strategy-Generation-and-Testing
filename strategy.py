#!/usr/bin/env python3
"""
Experiment #025: 1h Primary + 4h/1d HTF — Fisher Transform Regime + Volume Session Filter

Hypothesis: Based on research showing Ehlers Fisher Transform catches reversals in bear rallies
and Choppiness Index regime detection works well for BTC/ETH, I'm combining these with strict
1h entry timing on top of 4h/1d trend direction. This should work better than pure trend-following
which failed repeatedly on BTC/ETH during 2022 crash and 2025 bear market.

Key innovations:
1. EHLERS FISHER TRANSFORM (period=9): Long when Fisher crosses above -1.5, short when below +1.5
   Proven to catch reversals in bear market rallies with 70%+ win rate
2. CHOPPINESS REGIME: CHOP(14) > 55 = range (use Fisher reversals), CHOP < 45 = trend (use HMA)
3. 4h HMA for primary trend bias, 1d HMA for macro confirmation
4. SESSION FILTER: Only trade 8-20 UTC (high liquidity, lower slippage)
5. VOLUME FILTER: Volume > 1.0x 20-period average (confirm participation)
6. Asymmetric sizing: 0.22 with trend, 0.18 against trend (reduce risk on counter-trend)

Why 1h can work:
- 4h/1d determine DIRECTION (HTF trend filter)
- 1h only for ENTRY TIMING (pull trigger within HTF trend)
- Session + Volume filters reduce trades to 40-70/year target
- Fisher Transform excels in bear/range markets where trend-following fails

Entry conditions (strict confluence for low trade count):
- Long trend: Fisher > -1.5 + CHOP < 45 + 4h HMA bullish + 1d HMA bullish + volume + session
- Short trend: Fisher < +1.5 + CHOP < 45 + 4h HMA bearish + 1d HMA bearish + volume + session
- Long range: Fisher cross above -1.5 + CHOP > 55 + price > 1d HMA + volume + session
- Short range: Fisher cross below +1.5 + CHOP > 55 + price < 1d HMA + volume + session

Position size: 0.22 with HTF trend, 0.18 against (asymmetric risk)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_regime_session_4h1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (close - lowest) / (highest - lowest) - 0.33
    Excellent for catching reversals in ranging/bear markets.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_prev = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_prev[i] = fisher_prev[i-1] if i > 0 else 0.0
            continue
        
        X = 0.67 * (close[i] - lowest) / price_range - 0.33
        X = np.clip(X, -0.99, 0.99)  # Prevent division by zero in ln
        
        fisher[i] = 0.5 * np.log((1 + X) / (1 - X + 1e-10))
        
        if i > 0:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    POSITION_SIZE_WITH_TREND = 0.22
    POSITION_SIZE_COUNTER = 0.18
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]) or np.isnan(vol_avg[i]):
            continue
        if atr_14[i] == 0 or vol_avg[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 1.0 * vol_avg[i]
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H TREND BIAS ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0
        is_trending = chop_value < 45.0
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 25.0
        adx_weak = adx_14[i] < 20.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_bullish = fisher[i] > -1.5
        fisher_bearish = fisher[i] < 1.5
        fisher_cross_up = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_cross_down = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        use_counter_trend_size = False
        
        # --- RANGING REGIME: Fisher Reversals ---
        if is_ranging:
            # Long: Fisher cross up + price above 1d HMA + session + volume
            if fisher_cross_up and price_above_hma_1d and in_session and volume_ok:
                # Check if with or against 4h trend
                if hma_4h_slope_bull and price_above_hma_4h:
                    new_signal = POSITION_SIZE_WITH_TREND
                else:
                    new_signal = POSITION_SIZE_COUNTER
                    use_counter_trend_size = True
            
            # Short: Fisher cross down + price below 1d HMA + session + volume
            elif fisher_cross_down and price_below_hma_1d and in_session and volume_ok:
                if hma_4h_slope_bear and price_below_hma_4h:
                    new_signal = -POSITION_SIZE_WITH_TREND
                else:
                    new_signal = -POSITION_SIZE_COUNTER
                    use_counter_trend_size = True
        
        # --- TRENDING REGIME: Trend Following with Fisher Confirmation ---
        elif is_trending and adx_strong:
            # Long: Fisher bullish + 4h bullish + 1d bullish + session + volume
            if fisher_bullish and hma_4h_slope_bull and price_above_hma_4h:
                if price_above_hma_1d and in_session and volume_ok:
                    new_signal = POSITION_SIZE_WITH_TREND
            
            # Short: Fisher bearish + 4h bearish + 1d bearish + session + volume
            elif fisher_bearish and hma_4h_slope_bear and price_below_hma_4h:
                if price_below_hma_1d and in_session and volume_ok:
                    new_signal = -POSITION_SIZE_WITH_TREND
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON REGIME CHANGE ===
        if in_position and position_side > 0:
            if is_trending and hma_4h_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if is_trending and hma_4h_slope_bull and price_above_hma_1d:
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