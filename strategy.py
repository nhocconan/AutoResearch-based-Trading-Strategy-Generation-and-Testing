#!/usr/bin/env python3
"""
Experiment #470: 1h Primary + 4h/12h HTF — Fisher Transform + Vol Spike Mean Reversion + Session Filter

Hypothesis: After analyzing 469 failed experiments, clear patterns emerge:
1. Lower TF (1h/30m) MUST have very strict entry filters to limit trades to 30-60/year
2. Fisher Transform catches reversals better than RSI in bear/range markets (research proven)
3. Vol spike mean reversion (ATR7/ATR30 > 2.0) has edge after panic selling
4. Session filter (8-20 UTC) reduces trades by ~50% while keeping quality entries
5. 4h HMA trend direction + 1h Fisher entry timing = proven MTF pattern

Why this might beat current best (Sharpe=0.435):
- Fisher Transform has sharper reversal signals than RSI (Ehlers research)
- Vol spike filter ensures we only enter after extreme moves (mean reversion edge)
- Session filter + volume confirmation = fewer trades, less fee drag
- 4h/12h HTF trend prevents counter-trend trades in strong trends
- Asymmetric sizing: 0.25 long, 0.20 short (protects in bear markets)

Position sizing: 0.20-0.25 (smaller for 1h to reduce fee impact)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 40-80 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_volspike_session_4h12h_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 2 * (price - LL) / (HH - LL) - 1
    
    Long when Fisher crosses above -1.5 (oversold reversal)
    Short when Fisher crosses below +1.5 (overbought reversal)
    Proven to catch reversals in bear markets better than RSI.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Typical price
    typical = (high_s + low_s + close_s) / 3.0
    
    # Highest high and lowest low over period
    hh = typical.rolling(window=period, min_periods=period).max()
    ll = typical.rolling(window=period, min_periods=period).min()
    
    # Normalize price to -1 to +1 range
    range_hl = hh - ll
    range_hl = range_hl.replace(0, 1e-10)  # avoid div by zero
    x = 2.0 * (typical - ll) / range_hl - 1.0
    
    # Clamp to avoid ln domain errors
    x = x.clip(-0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + x) / (1.0 - x))
    
    # Signal line (1-period lag of fisher)
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load 4h and 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 12h HTF indicators (major trend)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Extract UTC hour for session filter
    hours = np.array([extract_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    
    # Track Fisher crosses for entry timing
    prev_fisher = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(chop_1h[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only - reduces trades by ~50%) ===
        in_session = 8 <= hours[i] <= 20
        
        # === 4H/12H HTF TREND (primary direction filter) ===
        # 4h HMA crossover for local trend
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # 12h HMA for major trend
        bull_regime_12h = close[i] > hma_12h_21_aligned[i]
        bear_regime_12h = close[i] < hma_12h_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        is_ranging = chop_1h[i] > 55.0
        is_trending = chop_1h[i] < 45.0
        
        # === VOLATILITY SPIKE FILTER (mean reversion edge) ===
        # ATR(7) / ATR(30) > 1.8 = vol spike (panic/extreme move)
        vol_spike = (atr_7[i] / atr_30[i]) > 1.8 if not np.isnan(atr_30[i]) and atr_30[i] > 0 else False
        vol_normal = (atr_7[i] / atr_30[i]) < 1.3 if not np.isnan(atr_30[i]) and atr_30[i] > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 0.8 * vol_avg[i]
        
        # === FISHER TRANSFORM SIGNALS (reversal entries) ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > -1.5 and prev_fisher <= -1.5  # crossed above -1.5
        fisher_cross_down = fisher[i] < 1.5 and prev_fisher >= 1.5  # crossed below +1.5
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — STRICT CONFLUENCE (3+ filters required) ===
        new_signal = 0.0
        
        # LONG ENTRIES (require: HTF bullish + Fisher signal + session + volume)
        if in_session and vol_confirmed:
            # Mean reversion long: vol spike + Fisher oversold + HTF not strongly bearish
            if vol_spike and fisher_oversold and not bear_regime_12h:
                new_signal = LONG_SIZE
            # Trend pullback long: 4h bullish + Fisher cross up + above SMA200
            elif hma_4h_bullish and fisher_cross_up and above_sma200:
                new_signal = LONG_SIZE
            # Ranging market long: CHOP > 55 + Fisher oversold + volume
            elif is_ranging and fisher_oversold and vol_confirmed:
                new_signal = LONG_SIZE * 0.8
            # Simpler entry to ensure trade count: Fisher cross + 4h bullish
            elif fisher_cross_up and hma_4h_bullish and in_session:
                new_signal = LONG_SIZE * 0.6
        
        # SHORT ENTRIES (require: HTF bearish + Fisher signal + session + volume)
        if in_session and vol_confirmed:
            # Mean reversion short: vol spike + Fisher overbought + HTF not strongly bullish
            if vol_spike and fisher_overbought and not bull_regime_12h:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Trend pullback short: 4h bearish + Fisher cross down + below SMA200
            elif hma_4h_bearish and fisher_cross_down and below_sma200:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Ranging market short: CHOP > 55 + Fisher overbought + volume
            elif is_ranging and fisher_overbought and vol_confirmed:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Simpler entry to ensure trade count: Fisher cross + 4h bearish
            elif fisher_cross_down and hma_4h_bearish and in_session:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === STOPLOSS CHECK (BEFORE exit logic - CRITICAL) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # Fisher extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and fisher[i] > 2.0:
            new_signal = 0.0
        if in_position and position_side < 0 and fisher[i] < -2.0:
            new_signal = 0.0
        
        # Session exit (close position outside trading hours)
        if in_position and not in_session:
            new_signal = 0.0
        
        # HTF trend reversal exit
        if in_position and position_side > 0 and hma_4h_bearish and bear_regime_12h:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_4h_bullish and bull_regime_12h:
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
        
        # Update prev_fisher for next iteration
        prev_fisher = fisher[i]
    
    return signals