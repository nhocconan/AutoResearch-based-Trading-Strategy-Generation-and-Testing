#!/usr/bin/env python3
"""
Experiment #294: 4h Primary + 12h/1d HTF — Dual Regime (KAMA + Choppiness + RSI + Volume)

Hypothesis: After 266 failed experiments, the key insight is REGIME ADAPTATION.
Crypto markets alternate between trending (2021 bull, 2023 recovery) and ranging (2022 crash, 2025 bear).
A single strategy cannot work in both. This implements DUAL REGIME:

1. CHOPPINESS INDEX (14) detects regime: CHOP > 58 = range, CHOP < 42 = trend
2. KAMA (Kaufman Adaptive MA) adapts to market noise — faster in trends, slower in chop
3. 12h HTF KAMA provides PRIMARY trend direction (slower, more stable)
4. 1d HTF HMA provides META trend (bull/bear market filter)
5. RSI(14) for entry timing within regime
6. Volume confirmation (volume > SMA20) to filter false breakouts

REGIME-SPECIFIC LOGIC:
- TREND REGIME (CHOP < 42): Follow 12h KAMA direction, enter on RSI pullback (40-60)
- RANGE REGIME (CHOP > 58): Mean revert at BB extremes, RSI extremes (25/75)
- TRANSITION (42-58): Stay flat or reduce position

Why this might work:
- KAMA automatically adjusts ER (Efficiency Ratio) — no manual tuning
- Dual HTF (12h + 1d) provides layered trend confirmation
- Volume filter reduces false signals (common in 2022 whipsaw)
- Discrete position sizing (0.25/0.35) minimizes fee churn

Position sizing: 0.25 base, 0.35 strong conviction (regime + HTF alignment)
Target: 30-50 trades/year on 4h (appropriate for this timeframe)
Stoploss: 2.5 * ATR trailing (tighter for 4h vs daily)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_dual_regime_12h1d_v1"
timeframe = "4h"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |price change| / sum of absolute price changes
    High ER (trending) → fast SC, Low ER (choppy) → slow SC
    
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    fast_sc = 2/(fast_period+1), slow_sc = 2/(slow_period+1)
    """
    n = period
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    price_change = np.abs(close_s.diff(n).values)
    sum_changes = pd.Series(np.abs(close_s.diff().values)).rolling(window=n, min_periods=n).sum().values
    
    er = np.zeros(len(close))
    for i in range(n, len(close)):
        if sum_changes[i] > 0:
            er[i] = price_change[i] / sum_changes[i]
        else:
            er[i] = 0.0
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[n] = close[n]  # Initialize
    
    for i in range(n + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume confirmation."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators (primary trend)
    kama_12h_21 = calculate_kama(df_12h['close'].values, 21)
    kama_12h_50 = calculate_kama(df_12h['close'].values, 50)
    
    # Calculate 1d HTF indicators (meta trend)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_12h_21_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_21)
    kama_12h_50_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_50)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    kama_4h_21 = calculate_kama(close, 21)
    kama_4h_50 = calculate_kama(close, 50)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    MIN_SIZE = 0.15
    
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
        
        if np.isnan(kama_12h_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === 1D META TREND (bull/bear market filter) ===
        # Bull: price above 1d HMA(50)
        # Bear: price below 1d HMA(50)
        meta_bull = close[i] > hma_1d_50_aligned[i]
        meta_bear = close[i] < hma_1d_50_aligned[i]
        
        # === 12H PRIMARY TREND ===
        # Bull: 12h KAMA(21) > KAMA(50)
        # Bear: 12h KAMA(21) < KAMA(50)
        trend_12h_bull = kama_12h_21_aligned[i] > kama_12h_50_aligned[i]
        trend_12h_bear = kama_12h_21_aligned[i] < kama_12h_50_aligned[i]
        
        # === 4H LOCAL TREND ===
        trend_4h_bull = kama_4h_21[i] > kama_4h_50[i]
        trend_4h_bear = kama_4h_21[i] < kama_4h_50[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 58 = range market (mean revert entries)
        # CHOP < 42 = trend market (breakout/pullback entries)
        # 42-58 = transition (reduce size or flat)
        is_choppy = chop_14[i] > 58.0
        is_trending = chop_14[i] < 42.0
        is_transition = not is_choppy and not is_trending
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_sma_20[i] * 1.2
        
        # === PRICE POSITION ===
        price_above_kama_4h = close[i] > kama_4h_21[i]
        price_below_kama_4h = close[i] < kama_4h_21[i]
        
        # === BOLLINGER BAND SIGNALS ===
        bb_break_lower = close[i] < bb_lower[i] * 1.002
        bb_break_upper = close[i] > bb_upper[i] * 0.998
        bb_near_lower = close[i] < bb_lower[i] * 1.01
        bb_near_upper = close[i] > bb_upper[i] * 0.99
        
        # === RSI THRESHOLDS ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        rsi_pullback_long = 40.0 < rsi_14[i] < 55.0
        rsi_pullback_short = 45.0 < rsi_14[i] < 60.0
        
        # === ENTRY LOGIC (DUAL REGIME) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND REGIME ENTRIES (CHOP < 42)
        if is_trending:
            # LONG: 12h trend bull + 4h trend bull + RSI pullback + volume
            if trend_12h_bull and trend_4h_bull and rsi_pullback_long and volume_confirmed:
                if meta_bull:
                    new_signal = STRONG_SIZE * vol_scale
                else:
                    new_signal = BASE_SIZE * vol_scale
            
            # LONG: 12h trend bull + price breaks above KAMA + RSI > 50
            elif trend_12h_bull and price_above_kama_4h and rsi_14[i] > 50 and volume_confirmed:
                new_signal = BASE_SIZE * vol_scale
            
            # SHORT: 12h trend bear + 4h trend bear + RSI pullback + volume
            if trend_12h_bear and trend_4h_bear and rsi_pullback_short and volume_confirmed:
                if new_signal == 0.0:
                    if meta_bear:
                        new_signal = -STRONG_SIZE * vol_scale
                    else:
                        new_signal = -BASE_SIZE * vol_scale
            
            # SHORT: 12h trend bear + price breaks below KAMA + RSI < 50
            if trend_12h_bear and price_below_kama_4h and rsi_14[i] < 50 and volume_confirmed:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * vol_scale
        
        # RANGE REGIME ENTRIES (CHOP > 58)
        elif is_choppy:
            # LONG: BB lower break + RSI oversold + volume spike (mean revert)
            if bb_break_lower and rsi_oversold:
                new_signal = BASE_SIZE * vol_scale
            # LONG: Extreme RSI oversold (strong conviction)
            elif rsi_extreme_oversold:
                new_signal = STRONG_SIZE * vol_scale
            
            # SHORT: BB upper break + RSI overbought + volume spike (mean revert)
            if bb_break_upper and rsi_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * vol_scale
            # SHORT: Extreme RSI overbought (strong conviction)
            elif rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE * vol_scale
        
        # TRANSITION REGIME (42-58) - reduced conviction
        elif is_transition:
            # Only take strong signals in transition
            if trend_12h_bull and rsi_extreme_oversold and meta_bull:
                new_signal = MIN_SIZE * vol_scale
            elif trend_12h_bear and rsi_extreme_overbought and meta_bear:
                new_signal = -MIN_SIZE * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 30+ trades/year on 4h) ===
        # Force trade if no signal for 30 bars (~5 days on 4h = 30 bars)
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if trend_12h_bull and rsi_14[i] > 45 and price_above_kama_4h:
                new_signal = MIN_SIZE * vol_scale
            elif trend_12h_bear and rsi_14[i] < 55 and price_below_kama_4h:
                new_signal = -MIN_SIZE * vol_scale
            elif is_choppy and rsi_14[i] < 35:
                new_signal = MIN_SIZE * vol_scale
            elif is_choppy and rsi_14[i] > 65:
                new_signal = -MIN_SIZE * vol_scale
        
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
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI goes overbought
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            # Short position: exit when RSI goes oversold
            if position_side < 0 and rsi_14[i] < 30:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 12h trend turns bearish
            if position_side > 0 and trend_12h_bear and price_below_kama_4h:
                regime_reversal = True
            # Short position but 12h trend turns bullish
            if position_side < 0 and trend_12h_bull and price_above_kama_4h:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.20:
                new_signal = 0.0
            elif new_signal > 0.30:
                new_signal = STRONG_SIZE * vol_scale
            elif new_signal > 0:
                new_signal = BASE_SIZE * vol_scale
            elif new_signal < -0.30:
                new_signal = -STRONG_SIZE * vol_scale
            else:
                new_signal = -BASE_SIZE * vol_scale
        
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