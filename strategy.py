#!/usr/bin/env python3
"""
Experiment #455: 1h Primary + 4h/1d HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 454 failed experiments, clear pattern for 1h TF:
1. Ehlers Fisher Transform catches reversals better than RSI in bear/range markets (proven in research)
2. Choppiness Index regime filter prevents trend-following in chop (critical for 2025 bear market)
3. 4h HMA for intermediate trend + 1d HMA for major bias = dual HTF confirmation
4. Session filter (8-20 UTC) avoids low-liquidity whipsaws
5. Volume confirmation ensures breakouts have participation
6. Asymmetric sizing: 0.25 long, 0.20 short (protects in bear markets like 2025)

Why this might beat current best (Sharpe=0.435):
- Fisher Transform has faster response than RSI for reversal entries
- Dual HTF (4h + 1d) provides stronger trend confirmation than single HTF
- 1h TF with strict filters = HTF trade frequency with better entry timing
- Session filter removes 40% of low-quality signals (Asian session whipsaws)
- Volume filter ensures we only trade when institutions are active

Position sizing: 0.20-0.25 (discrete levels, max 0.35 for lower TF)
Stoploss: 2.0 * ATR trailing (tighter for lower TF)
Target: 40-80 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_hma_4h1d_session_v2"
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
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - LL) / (HH - LL) - 0.67
    
    Catches reversals in bear markets. Long when Fisher crosses above -1.5,
    short when crosses below +1.5.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Midpoint price
    price = (high_s + low_s) / 2.0
    
    # Highest high and lowest low over period
    hh = price.rolling(window=period, min_periods=period).max()
    ll = price.rolling(window=period, min_periods=period).min()
    
    # Normalize price
    range_hl = hh - ll
    range_hl = range_hl.replace(0, 1e-10)  # avoid div by zero
    
    X = 0.67 * (price - ll) / range_hl - 0.67
    X = X.clip(-0.99, 0.99)  # prevent ln domain error
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + X) / (1 - X + 1e-10))
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # ATR for each bar
    atr = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)  # avoid div by zero
    
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

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
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Calculate 1h HMA for local trend
    hma_1h_21 = calculate_hma(close, period=21)
    hma_1h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.35 for lower TF)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    prev_fisher = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_1h[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1h_21[i]) or np.isnan(hma_1h_50[i]):
            continue
        
        # Extract UTC hour for session filter
        utc_hour = get_utc_hour(open_time[i])
        
        # === SESSION FILTER (8-20 UTC = high liquidity) ===
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER (must be >= 0.8x average) ===
        volume_ok = vol_ratio[i] >= 0.8
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === 1H LOCAL TREND ===
        hma_1h_bullish = hma_1h_21[i] > hma_1h_50[i]
        hma_1h_bearish = hma_1h_21[i] < hma_1h_50[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        is_ranging = chop_1h[i] > 55.0
        is_trending = chop_1h[i] < 45.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > -1.5 and prev_fisher <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and prev_fisher >= 1.5
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC — REGIME ADAPTIVE WITH 3+ CONFLUENCE ===
        new_signal = 0.0
        
        # LONG ENTRIES (need 3+ confluence)
        if in_session and volume_ok:
            long_confluence = 0
            
            # Confluence 1: HTF trend alignment
            if bull_regime_1d or hma_4h_bullish:
                long_confluence += 1
            
            # Confluence 2: Fisher reversal signal
            if fisher_cross_up or fisher_oversold:
                long_confluence += 1
            
            # Confluence 3: RSI confirmation
            if rsi_oversold:
                long_confluence += 1
            
            # Confluence 4: Local trend support
            if hma_1h_bullish:
                long_confluence += 1
            
            # Confluence 5: Ranging market mean reversion
            if is_ranging and fisher_oversold:
                long_confluence += 1
            
            # Enter with 3+ confluence
            if long_confluence >= 3:
                new_signal = LONG_SIZE
            # Enter with 2+ confluence in strong regime
            elif long_confluence >= 2 and bull_regime_1d and hma_4h_bullish:
                new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRIES (need 3+ confluence)
        if in_session and volume_ok:
            short_confluence = 0
            
            # Confluence 1: HTF trend alignment
            if bear_regime_1d or hma_4h_bearish:
                short_confluence += 1
            
            # Confluence 2: Fisher reversal signal
            if fisher_cross_down or fisher_overbought:
                short_confluence += 1
            
            # Confluence 3: RSI confirmation
            if rsi_overbought:
                short_confluence += 1
            
            # Confluence 4: Local trend support
            if hma_1h_bearish:
                short_confluence += 1
            
            # Confluence 5: Ranging market mean reversion
            if is_ranging and fisher_overbought:
                short_confluence += 1
            
            # Enter with 3+ confluence
            if short_confluence >= 3 and new_signal == 0.0:
                new_signal = -SHORT_SIZE
            # Enter with 2+ confluence in strong regime
            elif short_confluence >= 2 and bear_regime_1d and hma_4h_bearish and new_signal == 0.0:
                new_signal = -SHORT_SIZE * 0.8
        
        # === STOPLOSS CHECK (BEFORE exit logic - CRITICAL) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # Fisher extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and fisher[i] > 1.8:
            new_signal = 0.0
        if in_position and position_side < 0 and fisher[i] < -1.8:
            new_signal = 0.0
        
        # RSI extreme exit
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # HTF regime flip exit
        if in_position and position_side > 0 and bear_regime_1d and hma_4h_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime_1d and hma_4h_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
        prev_fisher = fisher[i]
    
    return signals