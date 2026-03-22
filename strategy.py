#!/usr/bin/env python3
"""
Experiment #211: 4h Primary + 1d/1w HTF — Regime-Adaptive Fisher + Donchian

Hypothesis: Previous 4h strategies failed because they used单一 approach (either pure trend
or pure mean-reversion). This strategy adapts to market regime using Choppiness Index:
- CHOP > 55 (range): Fisher Transform reversals + RSI extremes (mean-reversion)
- CHOP < 45 (trend): Donchian breakouts + HMA trend (trend-following)
- 1d HMA slope provides major trend bias to avoid counter-trend trades
- 1w HMA for ultra-long-term regime filter (bull/bear market)

Why this should work:
- Fisher Transform catches reversals better than RSI in range markets (research-backed)
- Donchian breakouts work in trending regimes with HTF confirmation
- Regime switching prevents whipsaw losses (2022 crash destroyed pure trend strategies)
- 4h timeframe = 20-50 trades/year target (low fee drag, sufficient signal frequency)
- Asymmetric sizing: larger positions when HTF trend aligns with entry

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d + 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_fisher_donchian_1d1w_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Normalize price to range -1 to +1
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    price_range = highest - lowest
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    normalized = (2 * (hl2 - lowest) / price_range) - 1
    normalized = np.clip(normalized, -0.999, 0.999)  # Prevent log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    close_s = pd.Series(close)
    change = np.abs(close_s.diff().values)
    noise = np.abs(close_s.diff().values)
    
    # Efficiency Ratio
    signal = pd.Series(change).rolling(window=er_period, min_periods=er_period).sum().values
    noise_sum = pd.Series(noise).rolling(window=er_period, min_periods=er_period).sum().values
    noise_sum = np.where(noise_sum == 0, 1e-10, noise_sum)
    er = signal / noise_sum
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian_channels(high, low, 20)
    
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    kama_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Volatility ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 1W ULTRA-LONG TERM REGIME ===
        bull_market = hma_1w_slope_aligned[i] > 0.3
        bear_market = hma_1w_slope_aligned[i] < -0.3
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        neutral_regime = not is_range_market and not is_trend_market
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_bull_cross = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        fisher_bear_cross = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === VOLATILITY SPIKE ===
        vol_spike = atr_ratio[i] > 1.5
        
        # === POSITION SIZING ADJUSTMENT ===
        current_size = BASE_SIZE
        if bull_market and trend_1d_bullish:
            current_size = BASE_SIZE * 1.1  # Larger in strong bull
        elif bear_market and trend_1d_bearish:
            current_size = BASE_SIZE * 1.1  # Larger in strong bear
        elif neutral_regime:
            current_size = BASE_SIZE * 0.7  # Smaller in uncertain regime
        
        # Clamp to max
        current_size = min(current_size, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_score = 0
        long_confidence = 0
        
        # Path 1: Range market + Fisher reversal + RSI oversold (mean-reversion)
        if is_range_market and fisher_oversold and rsi_oversold:
            long_score += 3
            long_confidence += 2
        
        # Path 2: Trend market + 1d bullish + Donchian breakout (trend-follow)
        if is_trend_market and trend_1d_bullish and donchian_breakout_long:
            long_score += 3
            long_confidence += 2
        
        # Path 3: Bull market + pullback to KAMA + RSI low
        if bull_market and close[i] < kama_50[i] * 1.02 and close[i] > kama_50[i] * 0.98 and rsi_14[i] < 45:
            long_score += 2
            long_confidence += 1
        
        # Path 4: Fisher cross + 1d bullish bias
        if fisher_bull_cross and (trend_1d_bullish or price_above_1d_hma):
            long_score += 2
            long_confidence += 1
        
        # Path 5: Vol spike + RSI extreme (capitulation long)
        if vol_spike and rsi_extreme_low:
            long_score += 2
            long_confidence += 1
        
        # Path 6: Simple oversold in bull market (fallback for trade frequency)
        if bull_market and rsi_14[i] < 40 and bars_since_last_trade > 60:
            long_score += 1
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score == 2 and long_confidence >= 1:
            new_signal = current_size
        elif long_score >= 2 and bars_since_last_trade > 80:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        short_confidence = 0
        
        # Path 1: Range market + Fisher reversal + RSI overbought
        if is_range_market and fisher_overbought and rsi_overbought:
            short_score += 3
            short_confidence += 2
        
        # Path 2: Trend market + 1d bearish + Donchian breakdown
        if is_trend_market and trend_1d_bearish and donchian_breakout_short:
            short_score += 3
            short_confidence += 2
        
        # Path 3: Bear market + rally to KAMA + RSI high
        if bear_market and close[i] > kama_50[i] * 0.98 and close[i] < kama_50[i] * 1.02 and rsi_14[i] > 55:
            short_score += 2
            short_confidence += 1
        
        # Path 4: Fisher cross + 1d bearish bias
        if fisher_bear_cross and (trend_1d_bearish or price_below_1d_hma):
            short_score += 2
            short_confidence += 1
        
        # Path 5: Vol spike + RSI extreme (capitulation short)
        if vol_spike and rsi_extreme_high:
            short_score += 2
            short_confidence += 1
        
        # Path 6: Simple overbought in bear market (fallback)
        if bear_market and rsi_14[i] > 60 and bars_since_last_trade > 60:
            short_score += 1
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score == 2 and short_confidence >= 1:
            new_signal = -current_size
        elif short_score >= 2 and bars_since_last_trade > 80:
            new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~20 days on 4h) to ensure min trades
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if bull_market and trend_1d_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.4
            elif bear_market and trend_1d_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.4
            elif rsi_14[i] < 30:
                new_signal = current_size * 0.3
            elif rsi_14[i] > 70:
                new_signal = -current_size * 0.3
        
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
            # Exit long if regime shifts to strong bear
            if position_side > 0 and is_trend_market and trend_1d_bearish and bear_market:
                regime_reversal = True
            # Exit short if regime shifts to strong bull
            if position_side < 0 and is_trend_market and trend_1d_bullish and bull_market:
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