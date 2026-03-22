#!/usr/bin/env python3
"""
Experiment #600: 1h Primary + 4h HTF — Simplified RSI + Choppiness + Funding Rate Contrarian

Hypothesis: #595 failed (Sharpe=-0.231) due to overly strict filters (CRSI extremes 15/85, session 8-20 UTC).
This version simplifies entry logic while maintaining trade frequency control:
1. RSI(14) instead of CRSI - simpler, more reliable mean reversion signal
2. RSI thresholds 25/75 (not 15/85) - ensures trades actually trigger
3. Remove session filter - crypto trades 24/7, session filter unnecessary
4. Add funding rate contrarian signal - proven edge for BTC/ETH in bear markets
5. Simpler CHOP regime: >50 = range (mean revert), <50 = trend (follow)
6. 4h HMA(21) for trend bias (proven in #591, #594 with Sharpe>0.4)
7. Position size: 0.25 discrete, stoploss 2.5 ATR trailing

Why this might beat Sharpe=0.520:
- Funding rate contrarian works especially well in 2022 crash and 2025 bear
- Simpler logic = fewer bugs, more consistent signals
- RSI(14) more stable than CRSI for 1h timeframe
- 4h trend filter prevents counter-trend mean reversion trades

Position sizing: 0.25 discrete (control drawdown on 1h)
Target: >=30 trades/symbol train, >=3 test, Sharpe > 0 all symbols
Trade frequency: 40-70/year via RSI extremes + HTF trend confluence
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_chop_funding_4h_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection.
    CHOP > 50 = choppy/range market (mean reversion works)
    CHOP < 50 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def load_funding_rate(symbol):
    """
    Load funding rate data from processed funding parquet files.
    Returns aligned funding rate array for contrarian signal.
    Funding > 0.03% = crowded long (short signal)
    Funding < -0.03% = crowded short (long signal)
    """
    try:
        symbol_map = {
            'BTCUSDT': 'btcusdt',
            'ETHUSDT': 'ethusdt',
            'SOLUSDT': 'solusdt'
        }
        symbol_lower = symbol_map.get(symbol, symbol.lower())
        funding_path = f"data/processed/funding/{symbol_lower}.parquet"
        df_funding = pd.read_parquet(funding_path)
        return df_funding['funding_rate'].values
    except:
        return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HTF HMA for trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Volume SMA for confirmation
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load funding rate for contrarian signal
    symbol = prices.get('symbol', 'BTCUSDT')
    funding_rates = load_funding_rate(symbol)
    if funding_rates is not None and len(funding_rates) >= n:
        funding_aligned = funding_rates[:n]
    else:
        funding_aligned = np.zeros(n)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(vol_sma20[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_chop_regime = chop_14[i] > 50.0
        is_trend_regime = chop_14[i] < 50.0
        
        # === 4H TREND BIAS ===
        bull_bias_4h = close[i] > hma_4h_21_aligned[i]
        bear_bias_4h = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA slope for trend strength
        hma_4h_slope_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_slope_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.7 * vol_sma20[i]
        
        # === FUNDING RATE CONTRARIAN ===
        funding_extreme_long = funding_aligned[i] > 0.0003  # >0.03%
        funding_extreme_short = funding_aligned[i] < -0.0003  # <-0.03%
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        
        if volume_confirmed:
            # --- CHOP REGIME: Mean Reversion (RSI extremes) ---
            if is_chop_regime:
                # Long: RSI < 30 (oversold) + 4h bull bias OR funding extreme short
                if rsi_14[i] < 30.0 and (bull_bias_4h or funding_extreme_short):
                    new_signal = POSITION_SIZE
                
                # Short: RSI > 70 (overbought) + 4h bear bias OR funding extreme long
                elif rsi_14[i] > 70.0 and (bear_bias_4h or funding_extreme_long):
                    new_signal = -POSITION_SIZE
            
            # --- TREND REGIME: Trend Following (RSI pullback + trend) ---
            elif is_trend_regime:
                # Long: RSI < 45 (pullback) + 4h bull + slope confirmed
                if rsi_14[i] < 45.0 and bull_bias_4h and hma_4h_slope_bull:
                    new_signal = POSITION_SIZE
                
                # Short: RSI > 55 (pullback) + 4h bear + slope confirmed
                elif rsi_14[i] > 55.0 and bear_bias_4h and hma_4h_slope_bear:
                    new_signal = -POSITION_SIZE
        
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
        
        # === EXIT CONDITIONS (regime flip or bias flip) ===
        if in_position and position_side > 0:
            if bear_bias_4h and rsi_14[i] > 55.0:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if bull_bias_4h and rsi_14[i] < 45.0:
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