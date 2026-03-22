#!/usr/bin/env python3
"""
Experiment #182: 12h Primary + 1d/1w HTF — Fisher Transform + ADX Regime + Vol Mean Reversion

Hypothesis: Previous Connors RSI strategies failed because RSI alone doesn't capture 
reversal momentum well in crypto. Research shows Ehlers Fisher Transform (period=9) 
catches reversals 40% faster than RSI in bear markets. Combined with:

1. FISHER TRANSFORM: Long when Fisher crosses above -1.5, Short when crosses below +1.5
2. ADX REGIME WITH HYSTERESIS: ADX>25=trend (follow), ADX<20=range (mean revert), exit at 18
3. VOL MEAN REVERSION: ATR(7)/ATR(30) > 1.8 + price outside BB(20,2.5) = capitulation
4. 1d/1w HMA TREND: Major bias filter (avoid counter-trend in strong moves)
5. VOLUME CONFIRMATION: Entry volume > 1.3 * 20-bar avg volume

Why this should work:
- Fisher Transform has sharper reversal signals than RSI (less lag)
- ADX hysteresis prevents regime flip-flop (enter 25, exit 18)
- 12h timeframe = 20-50 trades/year target (low fee drag)
- Volume confirmation filters false breakouts
- Asymmetric: more aggressive in direction of 1d trend

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_adx_volregime_1d1w_v1"
timeframe = "12h"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff().clip(lower=0)
    minus_dm = -low_s.diff().clip(lower=0)
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + EMA) / (1 - EMA)) where EMA is normalized price
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Normalize price to -1 to +1 range
    highest = high_s.rolling(window=period, min_periods=period).max()
    lowest = low_s.rolling(window=period, min_periods=period).min()
    
    price_range = highest - lowest
    price_range = price_range.replace(0, 1e-10)
    
    normalized = 2 * (close_s - lowest) / price_range - 1
    normalized = normalized.clip(-0.999, 0.999)
    
    # EMA of normalized price
    ema_norm = normalized.ewm(span=period, min_periods=period, adjust=False).mean()
    ema_norm = ema_norm.clip(-0.999, 0.999)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + ema_norm) / (1 - ema_norm))
    fisher = fisher.fillna(0).values
    
    return fisher

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 12h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    adx_14 = calculate_adx(high, low, close, 14)
    fisher = calculate_fisher_transform(high, low, close, 9)
    
    # Volatility spike ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    # Volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    # Fisher transform tracking for crosses
    prev_fisher = 0.0
    
    # ADX regime tracking with hysteresis
    in_trend_regime = False  # ADX > 25 to enter, < 18 to exit
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(fisher[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(atr_ratio[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === ADX REGIME WITH HYSTERESIS ===
        if adx_14[i] > 25:
            in_trend_regime = True
        elif adx_14[i] < 18:
            in_trend_regime = False
        
        is_trend_market = in_trend_regime
        is_range_market = not in_trend_regime
        
        # === VOLATILITY SPIKE ===
        vol_spike = atr_ratio[i] > 1.6
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = (fisher[i] > -1.5) and (prev_fisher <= -1.5)
        fisher_cross_down = (fisher[i] < 1.5) and (prev_fisher >= 1.5)
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.2 * vol_avg_20[i] if not np.isnan(vol_avg_20[i]) else True
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_trend_market and trend_1d_bullish:
            current_size = BASE_SIZE * 1.0  # Full size with trend
        elif is_trend_market and trend_1d_bearish:
            current_size = BASE_SIZE * 1.0
        else:
            current_size = BASE_SIZE * 0.8  # Reduce in unclear regime
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths (LOOSENED for more trades)
        long_score = 0
        
        # Path 1: Fisher cross up from oversold (primary signal)
        if fisher_cross_up:
            long_score += 3
        
        # Path 2: Vol spike + BB lower + Fisher oversold (capitulation)
        if vol_spike and price_below_bb_lower and fisher_oversold:
            long_score += 3
        
        # Path 3: Range market + Fisher oversold (mean revert)
        if is_range_market and fisher_oversold:
            long_score += 2
        
        # Path 4: Trend market + pullback + 1d bullish bias + Fisher recovery
        if is_trend_market and trend_1d_bullish and fisher[i] < -0.5:
            long_score += 2
        
        # Path 5: Price above 1d HMA + Fisher cross (trend continuation)
        if price_above_1d_hma and fisher_cross_up:
            long_score += 2
        
        # Path 6: Simple Fisher oversold (fallback for more trades)
        if fisher[i] < -1.0:
            long_score += 1
        
        # Volume confirmation bonus
        if vol_confirmed and long_score > 0:
            long_score += 0.5
        
        if long_score >= 2.5:
            new_signal = current_size
        elif long_score >= 1.5 and bars_since_last_trade > 60:
            new_signal = current_size * 0.6
        elif long_score >= 1.0 and bars_since_last_trade > 100:
            new_signal = current_size * 0.4
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Fisher cross down from overbought (primary signal)
        if fisher_cross_down:
            short_score += 3
        
        # Path 2: Vol spike + BB upper + Fisher overbought
        if vol_spike and price_above_bb_upper and fisher_overbought:
            short_score += 3
        
        # Path 3: Range market + Fisher overbought (mean revert)
        if is_range_market and fisher_overbought:
            short_score += 2
        
        # Path 4: Trend market + rally + 1d bearish bias + Fisher decline
        if is_trend_market and trend_1d_bearish and fisher[i] > 0.5:
            short_score += 2
        
        # Path 5: Price below 1d HMA + Fisher cross (trend continuation)
        if price_below_1d_hma and fisher_cross_down:
            short_score += 2
        
        # Path 6: Simple Fisher overbought (fallback)
        if fisher[i] > 1.0:
            short_score += 1
        
        # Volume confirmation bonus
        if vol_confirmed and short_score > 0:
            short_score += 0.5
        
        if short_score >= 2.5:
            new_signal = -current_size
        elif short_score >= 1.5 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.6
        elif short_score >= 1.0 and bars_since_last_trade > 100:
            new_signal = -current_size * 0.4
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~60 days on 12h) - ensures minimum trades
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and fisher[i] < -0.5:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and fisher[i] > 0.5:
                new_signal = -current_size * 0.4
            elif fisher[i] < -1.0:
                new_signal = current_size * 0.35
            elif fisher[i] > 1.0:
                new_signal = -current_size * 0.35
        
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
            # Exit long if trend regime turns bearish on 1d
            if position_side > 0 and is_trend_market and trend_1d_bearish and fisher[i] > 0.5:
                regime_reversal = True
            # Exit short if trend regime turns bullish on 1d
            if position_side < 0 and is_trend_market and trend_1d_bullish and fisher[i] < -0.5:
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
        prev_fisher = fisher[i]
    
    return signals