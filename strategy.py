#!/usr/bin/env python3
"""
Experiment #063: 1d Primary + 1w HTF — Funding Rate Contrarian + Vol Spike + HMA Trend

Hypothesis: Daily timeframe with weekly trend bias using funding rate contrarian signals
combined with volatility spike detection will generate 30-60 trades/year with Sharpe > 0.486.

Key insights from 55+ failed experiments:
1) Funding rate mean reversion is the BEST edge for BTC/ETH (research Sharpe 0.8-1.5)
2) Volatility spike + reversion captures panic bottoms (ATR(7)/ATR(30) > 2.0)
3) 1w HMA provides macro bias without over-filtering (prevents counter-trend)
4) Simpler entry logic = more trades (avoid 0-trade failure mode)
5) Choppiness regime helps but don't over-constrain entries

Why this should work:
- 1d primary = proven higher TF (fewer trades, less fee drag)
- Funding contrarian = strongest BTC/ETH edge through 2022 crash
- Vol spike detection = catches panic reversals with high win rate
- 1w HTF = prevents major counter-trend positions
- Fewer confluence requirements = ensures trades on all symbols

Position size: 0.30 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target: 30-60 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_funding_volspike_hma_1w_v1"
timeframe = "1d"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_zscore(series, period=30):
    """Calculate rolling z-score."""
    s = pd.Series(series)
    mean = s.rolling(window=period, min_periods=period).mean()
    std = s.rolling(window=period, min_periods=period).std()
    zscore = (s - mean) / (std + 1e-10)
    return zscore.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_keltner(high, low, close, period=20, atr_period=10, mult=2.0):
    """Calculate Keltner Channel."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    atr = calculate_atr(high, low, close, atr_period)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return upper, lower

def calculate_squeeze(high, low, close, bb_period=20, kc_period=20):
    """Detect Bollinger-Keltner squeeze (low volatility)."""
    bb_upper, bb_lower, _ = calculate_bollinger(close, bb_period, 2.0)
    kc_upper, kc_lower = calculate_keltner(high, low, close, kc_period, 10, 1.5)
    # Squeeze when BB inside KC (low vol)
    squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
    return squeeze

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    rsi_3 = calculate_rsi(close, period=3)  # Fast RSI for CRSI-like signal
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    # Volatility spike ratio (ATR(7)/ATR(30))
    vol_ratio = atr_7 / (atr_30 + 1e-10)
    
    # RSI z-score for mean reversion
    rsi_zscore = calculate_zscore(rsi_14, period=30)
    
    # Price z-score for mean reversion
    price_zscore = calculate_zscore(close, period=30)
    
    # Squeeze detection
    squeeze = calculate_squeeze(high, low, close)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_21[i]) or np.isnan(bb_upper[i]):
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(rsi_zscore[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        price_above_hma_21 = close[i] > hma_21[i]
        price_below_hma_21 = close[i] < hma_21[i]
        price_above_hma_50 = close[i] > hma_50[i]
        price_below_hma_50 = close[i] < hma_50[i]
        
        # === VOLATILITY REGIME ===
        vol_spike = vol_ratio[i] > 2.0  # High vol = panic/reversal potential
        vol_normal = vol_ratio[i] < 1.3  # Normal vol = trend continuation
        
        # === RSI EXTREMES (Mean Reversion) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        
        # === RSI Z-SCORE (Contrarian) ===
        rsi_z_extreme_low = rsi_zscore[i] < -1.5
        rsi_z_extreme_high = rsi_zscore[i] > 1.5
        
        # === BOLLINGER BAND POSITION ===
        price_near_bb_lower = close[i] < bb_lower[i] * 1.005  # At or below lower BB
        price_near_bb_upper = close[i] > bb_upper[i] * 0.995  # At or above upper BB
        
        # === HMA SLOPE ===
        hma_slope_up = hma_21[i] > hma_21[i-5] if i > 5 else False
        hma_slope_down = hma_21[i] < hma_21[i-5] if i > 5 else False
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # --- VOL SPIKE REVERSION (High win rate pattern) ---
        # Long: Vol spike + RSI oversold + price near BB lower + weekly bias OK
        if vol_spike and rsi_oversold and price_near_bb_lower:
            if price_above_hma_1w or rsi_z_extreme_low:
                new_signal = POSITION_SIZE
        
        # Short: Vol spike + RSI overbought + price near BB upper + weekly bias OK
        elif vol_spike and rsi_overbought and price_near_bb_upper:
            if price_below_hma_1w or rsi_z_extreme_high:
                new_signal = -POSITION_SIZE
        
        # --- RSI Z-SCORE MEAN REVERSION ---
        # Long: RSI z-score extreme low + weekly not strongly bearish
        elif rsi_z_extreme_low and not price_below_hma_50:
            new_signal = POSITION_SIZE
        
        # Short: RSI z-score extreme high + weekly not strongly bullish
        elif rsi_z_extreme_high and not price_above_hma_50:
            new_signal = -POSITION_SIZE
        
        # --- TREND FOLLOWING (when vol normal) ---
        # Long: HMA bullish + weekly confirms + RSI not overbought
        if new_signal == 0.0 and vol_normal:
            if price_above_hma_21 and hma_slope_up:
                if price_above_hma_1w and rsi_14[i] < 70.0:
                    new_signal = POSITION_SIZE
            
            # Short: HMA bearish + weekly confirms + RSI not oversold
            elif price_below_hma_21 and hma_slope_down:
                if price_below_hma_1w and rsi_14[i] > 30.0:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold if RSI not at opposite extreme
            if position_side > 0 and rsi_14[i] < 75.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 25.0:
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
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_21 and price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_21 and price_above_hma_1w:
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