#!/usr/bin/env python3
"""
Experiment #011: 4h Primary + 1d/1w HTF — Adaptive KAMA Donchian Breakout

Hypothesis: Previous strategies failed due to overly restrictive entry conditions.
This strategy uses ADAPTIVE regime switching with looser thresholds to ensure
20-50 trades/year while maintaining edge:

1. 1w HMA(21) for MAJOR market regime (bull/bear)
2. 1d HMA(21) for INTERMEDIATE trend bias
3. 4h KAMA(14) for adaptive trend (responds to volatility changes)
4. ADX(14) for trend strength confirmation (>18 = trending)
5. Donchian(20) for breakout levels
6. Choppiness Index(14) for regime detection
7. RSI(14) for mean-reversion entries in choppy markets

REGIME SWITCHING:
- CHOP < 45 + ADX > 18 = TREND MODE (Donchian breakout with HTF bias)
- CHOP > 55 = RANGE MODE (RSI mean-reversion at Donchian bounds)
- 45 <= CHOP <= 55 = NO TRADE (unclear regime)

Why this should work:
- KAMA adapts to volatility (better than EMA in crypto)
- Dual regime captures both trending and ranging markets
- Looser ADX threshold (18 vs 25) ensures more trades
- 1w filter prevents counter-major-trend trades
- Discrete sizing (0.25/0.30) minimizes fee churn
- 2.5 ATR trailing stop protects capital

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 (trend), 0.20 (mean-reversion)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_donchian_adaptive_1d1w_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    
    Adapts to market volatility:
    - Fast SC (0.6667) in trending markets
    - Slow SC (0.02) in choppy markets
    
    ER = |Close - Close_n| / Sum(|Close - Close_prev|)
    SC = (ER * (fast - slow) + slow)^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    signal = np.abs(close - np.roll(close, er_period))
    signal[0:er_period] = np.nan
    
    noise = np.abs(close - np.roll(close, 1))
    noise[0] = 0
    
    noise_sum = pd.Series(noise).rolling(window=er_period, min_periods=er_period).sum().values
    er = signal / noise_sum
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    
    Measures trend strength (not direction):
    - ADX < 20 = weak/no trend
    - ADX 20-25 = developing trend
    - ADX > 25 = strong trend
    - ADX > 40 = very strong trend
    """
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
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[(plus_dm < 0) | (plus_dm <= minus_dm)] = 0
    minus_dm[(minus_dm < 0) | (minus_dm <= plus_dm)] = 0
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / np.maximum(atr, 1e-10)
    minus_di = 100 * minus_dm_s / np.maximum(atr, 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Upper = Highest High over period
    Lower = Lowest Low over period
    Middle = (Upper + Lower) / 2
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    
    return upper, lower, middle

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index
    
    CHOP > 61.8 = range/choppy (mean-reversion favorable)
    CHOP < 38.2 = trending (trend-following favorable)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of TR over period
    atr1_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # ATR(period)
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max().values - low_s.rolling(window=period, min_periods=period).min().values
    
    # Choppiness Index
    chop = 100 * (atr1_sum / np.maximum(atr_period, 1e-10)) / np.maximum(hh_ll, 1e-10) * np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_14 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx_14 = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, 20)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels)
    SIZE_TREND = 0.30
    SIZE_RANGE = 0.20
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(kama_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop[i]):
            continue
        
        # === 1W MAJOR REGIME ===
        # Price above 1w HMA = bull market (prefer longs)
        # Price below 1w HMA = bear market (prefer shorts)
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND BIAS ===
        trend_1d_bull = close[i] > hma_1d_21_aligned[i]
        trend_1d_bear = close[i] < hma_1d_21_aligned[i]
        
        # === 4H KAMA TREND ===
        kama_bull = close[i] > kama_14[i]
        kama_bear = close[i] < kama_14[i]
        
        # KAMA slope (trend direction)
        kama_slope_bull = kama_14[i] > kama_14[i-1] if i > 0 else False
        kama_slope_bear = kama_14[i] < kama_14[i-1] if i > 0 else False
        
        # === ADX TREND STRENGTH ===
        # Lowered threshold from 25 to 18 for more trades
        is_trending = adx_14[i] > 18
        is_weak = adx_14[i] <= 18
        
        # === CHOPPINESS REGIME ===
        # CHOP < 45 = trending (use breakout)
        # CHOP > 55 = range (use mean-reversion)
        # 45-55 = no trade (unclear)
        is_trend_regime = chop[i] < 45
        is_range_regime = chop[i] > 55
        
        # === DONCHIAN BREAKOUT ===
        breakout_upper = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_lower = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND MODE: Donchian breakout with HTF confirmation
        if is_trend_regime and is_trending:
            # LONG: 1w bull + 1d bull + KAMA bull + breakout upper
            if regime_bull and trend_1d_bull and kama_bull and breakout_upper:
                new_signal = SIZE_TREND
            
            # SHORT: 1w bear + 1d bear + KAMA bear + breakout lower
            if regime_bear and trend_1d_bear and kama_bear and breakout_lower:
                new_signal = -SIZE_TREND
        
        # RANGE MODE: Mean-reversion at Donchian bounds with RSI filter
        if is_range_regime:
            # LONG: Price at lower band + RSI oversold
            if close[i] <= donchian_lower[i] * 1.002 and rsi_14[i] < 40:
                # Only if not in strong bear regime
                if not (regime_bear and trend_1d_bear):
                    new_signal = SIZE_RANGE
            
            # SHORT: Price at upper band + RSI overbought
            if close[i] >= donchian_upper[i] * 0.998 and rsi_14[i] > 60:
                # Only if not in strong bull regime
                if not (regime_bull and trend_1d_bull):
                    new_signal = -SIZE_RANGE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~50 days on 4h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            # Weaker trend entry
            if trend_1d_bull and kama_bull and adx_14[i] > 15:
                new_signal = SIZE_TREND * 0.7
            elif trend_1d_bear and kama_bear and adx_14[i] > 15:
                new_signal = -SIZE_TREND * 0.7
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1d_bear and kama_bear:
                trend_reversal = True
            if position_side < 0 and trend_1d_bull and kama_bull:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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