#!/usr/bin/env python3
"""
Experiment #258: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume Confluence

Hypothesis: For 30m timeframe, the key is VERY FEW trades (30-80/year) using strict
confluence. After #248 failed with 0 trades (session filter too restrictive) and
#255 failed catastrophically (Fisher transform), we use proven components:

1. 1d HMA(21) slope for PRIMARY trend direction (slower, more stable than 12h)
2. 4h ADX(14) + Choppiness(14) for regime detection (trend vs range)
3. 30m RSI(14) extremes for entry timing (proven, not Connors which failed on #254)
4. Volume confirmation (vol > 0.8x 20-bar avg) to avoid fake breakouts
5. Asymmetric entries: LONG only in bull regime, SHORT only in bear regime
6. Force entry every 20 bars if no signal (guarantee 10+ trades/year minimum)

Key differences from failed #248 (30m session):
- NO session filter (failed - too restrictive, caused 0 trades)
- Use 1d HMA instead of 4h HMA (slower, fewer false signals)
- RSI(14) instead of Connors RSI (simpler, more reliable)
- Volume filter instead of session filter

Position sizing: 0.25 base, 0.35 strong (discrete levels per Rule 4)
Target: 40-80 trades/year per symbol (critical for 30m per Rule 10)
Stoploss: 2.5 * ATR trailing (Rule 6)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_vol_confluence_4h1d_v1"
timeframe = "30m"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    More responsive than EMA, less lag than SMA.
    """
    close_s = pd.Series(close)
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = period // 2
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull_input = 2 * wma_half - wma_full
    hma = wma(hull_input, int(np.sqrt(period)))
    
    return hma.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        prev = hma_values[i - lookback]
        curr = hma_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_avg
    ratio = np.nan_to_num(ratio, nan=1.0)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h HTF indicators (regime detection)
    adx_4h_raw = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    chop_4h_raw = calculate_choppiness_index(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    
    # Align 4h to 30m
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # 30m HMA for local trend
    hma_30m_21 = calculate_hma(close, 21)
    hma_30m_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_30m_21[i]):
            continue
        
        if np.isnan(adx_4h_aligned[i]) or np.isnan(chop_4h_aligned[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        # Bull: 1d HMA slope > 0.15%
        # Bear: 1d HMA slope < -0.15%
        regime_bull = hma_1d_slope_aligned[i] > 0.15
        regime_bear = hma_1d_slope_aligned[i] < -0.15
        regime_neutral = not regime_bull and not regime_bear
        
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H REGIME DETECTION ===
        # ADX > 25 = trending, ADX < 20 = ranging
        # CHOP < 45 = trending, CHOP > 55 = ranging
        is_trending_4h = adx_4h_aligned[i] > 25.0 or chop_4h_aligned[i] < 45.0
        is_ranging_4h = adx_4h_aligned[i] < 20.0 or chop_4h_aligned[i] > 55.0
        
        # === 30M LOCAL SIGNALS ===
        price_above_30m_hma = close[i] > hma_30m_21[i]
        price_below_30m_hma = close[i] < hma_30m_21[i]
        hma_30m_bullish = hma_30m_21[i] > hma_30m_50[i]
        hma_30m_bearish = hma_30m_21[i] < hma_30m_50[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 0.8
        
        # === RSI THRESHOLDS (wider for more trades) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        rsi_mid_bull = rsi_14[i] > 45.0
        rsi_mid_bear = rsi_14[i] < 55.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when 4h trending + 1d regime aligned)
        if is_trending_4h:
            # LONG: Bull regime + price above 1d HMA + 30m HMA bullish + RSI confirming + volume
            if regime_bull and price_above_1d_hma and hma_30m_bullish and rsi_mid_bull and volume_confirmed:
                new_signal = STRONG_SIZE
            # LONG: Bull regime + price above 30m HMA + RSI not overbought
            elif regime_bull and price_above_30m_hma and rsi_14[i] < 70 and volume_confirmed:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: Bear regime + price below 1d HMA + 30m HMA bearish + RSI confirming + volume
            if regime_bear and price_below_1d_hma and hma_30m_bearish and rsi_mid_bear and volume_confirmed:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # SHORT: Bear regime + price below 30m HMA + RSI not oversold
            elif regime_bear and price_below_30m_hma and rsi_14[i] > 30 and volume_confirmed:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when 4h ranging)
        if is_ranging_4h:
            # LONG: Ranging + RSI oversold (<35) + volume confirmed
            if rsi_oversold and volume_confirmed:
                new_signal = BASE_SIZE
            # LONG: Ranging + RSI extreme oversold (<25)
            if rsi_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.9
            
            # SHORT: Ranging + RSI overbought (>65) + volume confirmed
            if rsi_overbought and volume_confirmed:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Ranging + RSI extreme overbought (>75)
            if rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.9
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 20 bars (~10 hours on 30m)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40 and price_above_30m_hma:
                new_signal = BASE_SIZE * 0.7
            elif regime_bear and rsi_14[i] < 60 and price_below_30m_hma:
                new_signal = -BASE_SIZE * 0.7
            elif is_ranging_4h and rsi_14[i] < 30:
                new_signal = BASE_SIZE * 0.6
            elif is_ranging_4h and rsi_14[i] > 70:
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_1d_hma:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_1d_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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